// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP64 arithmetic with split-word VGPR operands and guest FP policies.
// Owns public entry points and their validation/dispatch policy.

#include "fp/fp64.h"
#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace {

// Return instruction modifiers with OMOD disabled when RDNA3 guest mode requires it.
uint64_t rdna3_output_modifiers(uint64_t flags, uint64_t mode) {
  return !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)
             ? mode & ~uint64_t(RJ_ARITH_ALU_OMOD_HALF)
             : mode;
}

// Evaluate FP64 arithmetic for the selected lanes. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::Fp64 Op, bool Rdna3 = false>
int arithmetic(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  uint32_t known =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  if constexpr (rocjitsu::arith::fp64_sources(Op) >= 2) {
    known |= RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_B;
  }
  if constexpr (rocjitsu::arith::fp64_sources(Op) == 3) {
    known |= RJ_ARITH_ALU_ABS_C | RJ_ARITH_ALU_NEG_C;
  }

  constexpr uint64_t fp_flags = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE |
                                (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0);
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, false, fp_flags)) {
    return error;
  }
  if constexpr (Rdna3) {
    mode = uint32_t(rdna3_output_modifiers(flags, mode));
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::fp64_x86_64_v3(Op, flags, exec_mask, mode, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[2][32];
  for (int lane = 0; lane < 32; ++lane) {
    double x = rocjitsu::arith::fp64_input(a, lane, mode), y = 0;
    if constexpr (rocjitsu::arith::fp64_sources(Op) >= 2) {
      y = rocjitsu::arith::fp64_input(b, lane, mode >> 1);
    }
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      x = rocjitsu::arith::fp64_flush_denorm(x);
      y = rocjitsu::arith::fp64_flush_denorm(y);
    }
    double value;

    if constexpr (Op == rocjitsu::arith::Fp64::Fma) {
      double z = rocjitsu::arith::fp64_input(c, lane, mode >> 2);
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        z = rocjitsu::arith::fp64_flush_denorm(z);
      }
      value = std::fma(x, y, z);
    }

    if constexpr (Op == rocjitsu::arith::Fp64::Trunc) {
      value = std::trunc(x);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Ceil) {
      value = std::ceil(x);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Rndne) {
      value = rocjitsu::arith::fp64_rndne(x);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Floor) {
      value = std::floor(x);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Sqrt) {
      value = std::sqrt(x);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Rcp) {
      value = 1.0 / x;
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Rsq) {
      value = 1.0 / std::sqrt(x);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Fract) {
      value = x - std::floor(x);
      double limit = rocjitsu::arith::as_double(0x3fefffffffffffffULL);
      if (value > limit) {
        value = limit;
      }
    }
    const bool preserve_nan =
        (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) &&
        (rocjitsu::arith::double_bits(value) & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL;
    value = rocjitsu::arith::fp64_output(value, preserve_nan ? mode & ~RJ_ARITH_ALU_CLAMP : mode);
    if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
      value = rocjitsu::arith::fp64_flush_denorm(value);
    }
    uint64_t bits = rocjitsu::arith::double_bits(value);
    result[0][lane] = uint32_t(bits);
    result[1][lane] = uint32_t(bits >> 32);
  }
  // Stage both halves before any destination write, including cross-half aliases.
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[reg][lane] = result[reg][lane];
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate FP64 arithmetic for Wave64. Reject invalid controls before changing outputs.
template <rocjitsu::arith::Fp64 Op, bool Rdna3 = false>
int arithmetic_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
                      const uint32_t *const *a, const uint32_t *const *b,
                      const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = arithmetic<Op, Rdna3>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint32_t *dest[] = {d[0] + half * 32, d[1] + half * 32};
    const uint32_t *x[] = {a[0] + half * 32, a[1] + half * 32};
    const uint32_t *y[2] = {}, *z[2] = {};
    if constexpr (rocjitsu::arith::fp64_sources(Op) >= 2) {
      for (int word = 0; word < 2; ++word) {
        y[word] = b[word] + half * 32;
      }
    }
    if constexpr (rocjitsu::arith::fp64_sources(Op) == 3) {
      for (int word = 0; word < 2; ++word) {
        z[word] = c[word] + half * 32;
      }
    }
    if (int error =
            arithmetic<Op, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), mode, dest, x, y, z)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Fma, true>(flags, exec_mask, mode, d, a, b, c);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Fma>(flags, exec_mask, mode, d, a, b, c);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Trunc>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Ceil>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Rndne>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Floor>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Fract>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Sqrt>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Rcp>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Rsq>(flags, exec_mask, mode, d, a, nullptr, nullptr);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Fma, true>(flags, exec_mask, mode, d, a, b, c);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fma_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Fma>(flags, exec_mask, mode, d, a, b, c);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Trunc>(flags, exec_mask, mode, d, a, nullptr,
                                                         nullptr);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Ceil>(flags, exec_mask, mode, d, a, nullptr,
                                                        nullptr);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Rndne>(flags, exec_mask, mode, d, a, nullptr,
                                                         nullptr);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Floor>(flags, exec_mask, mode, d, a, nullptr,
                                                         nullptr);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Fract>(flags, exec_mask, mode, d, a, nullptr,
                                                         nullptr);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Sqrt>(flags, exec_mask, mode, d, a, nullptr,
                                                        nullptr);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Rcp>(flags, exec_mask, mode, d, a, nullptr,
                                                       nullptr);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Rsq>(flags, exec_mask, mode, d, a, nullptr,
                                                       nullptr);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Trunc, true>(flags, exec_mask, mode, d, a, nullptr,
                                                        nullptr);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Trunc, true>(flags, exec_mask, mode, d, a,
                                                               nullptr, nullptr);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Ceil, true>(flags, exec_mask, mode, d, a, nullptr,
                                                       nullptr);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Ceil, true>(flags, exec_mask, mode, d, a, nullptr,
                                                              nullptr);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Rndne, true>(flags, exec_mask, mode, d, a, nullptr,
                                                        nullptr);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Rndne, true>(flags, exec_mask, mode, d, a,
                                                               nullptr, nullptr);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Floor, true>(flags, exec_mask, mode, d, a, nullptr,
                                                        nullptr);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Floor, true>(flags, exec_mask, mode, d, a,
                                                               nullptr, nullptr);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Fract, true>(flags, exec_mask, mode, d, a, nullptr,
                                                        nullptr);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                             const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Fract, true>(flags, exec_mask, mode, d, a,
                                                               nullptr, nullptr);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Sqrt, true>(flags, exec_mask, mode, d, a, nullptr,
                                                       nullptr);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Sqrt, true>(flags, exec_mask, mode, d, a, nullptr,
                                                              nullptr);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Rcp, true>(flags, exec_mask, mode, d, a, nullptr,
                                                      nullptr);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Rcp, true>(flags, exec_mask, mode, d, a, nullptr,
                                                             nullptr);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic<rocjitsu::arith::Fp64::Rsq, true>(flags, exec_mask, mode, d, a, nullptr,
                                                      nullptr);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return arithmetic_wave64<rocjitsu::arith::Fp64::Rsq, true>(flags, exec_mask, mode, d, a, nullptr,
                                                             nullptr);
}
