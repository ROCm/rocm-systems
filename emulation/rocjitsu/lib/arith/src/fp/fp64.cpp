// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP64 arithmetic and selection with split-word VGPR operands and guest FP policies.
// Also implements legacy, number-preferring and NaN-propagating min/max selection.
// Owns public entry points and their validation/dispatch policy.

#include "common/alu.h"
#include "common/internal.h"
#include "fp/float_output.h"
#include "fp/fp64_impl.h"
#include "fp/fp64_round.h"
#include "fp/minmax_impl.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Evaluate FP64 arithmetic or selection for the selected lanes. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::Fp64 Op, bool Rdna3 = false>
int fp64(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
         const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  uint32_t known =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;
  if constexpr (rocjitsu::arith::fp64_sources(Op) >= 2) {
    known |= RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_B;
  }
  if constexpr (rocjitsu::arith::fp64_sources(Op) == 3) {
    known |= RJ_ARITH_ALU_ABS_C | RJ_ARITH_ALU_NEG_C;
  }

  constexpr uint64_t fp_flags =
      RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
      RJ_ARITH_FP_IEEE_MODE |
      ((Op != rocjitsu::arith::Fp64::Rcp && Op != rocjitsu::arith::Fp64::Rsq &&
        Op != rocjitsu::arith::Fp64::Sqrt)
           ? RJ_ARITH_FP_ROUND_MASK
           : 0) |
      (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0);
  if (int error = rocjitsu::arith::validate(flags, instruction_flags & ~known, fp_flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (!rocjitsu::arith::is_selection(Op) &&
      (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if constexpr (Rdna3) {
    instruction_flags = uint32_t(rocjitsu::arith::rdna3_output_modifiers(flags, instruction_flags));
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if constexpr (Op == rocjitsu::arith::Fp64::Fma || Op == rocjitsu::arith::Fp64::Add ||
                Op == rocjitsu::arith::Fp64::Mul) {
    if ((flags & RJ_ARITH_FP_ROUND_MASK) &&
        rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
      rocjitsu::arith::fp64_directed_x86_64_v4(Op, flags, exec_mask, instruction_flags, d, a, b, c);
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  // AVX2 has no per-instruction directed FP64 ADD/MUL/FMA. Use the inherited integer
  // model for directed modes; nearest rounding retains the existing SIMD path.
  constexpr bool directed_arithmetic =
      Op == rocjitsu::arith::Fp64::Fma || Op == rocjitsu::arith::Fp64::Add ||
      Op == rocjitsu::arith::Fp64::Mul || Op == rocjitsu::arith::Fp64::Fract;
  if ((!directed_arithmetic || !(flags & RJ_ARITH_FP_ROUND_MASK)) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::fp64_x86_64_v3(Op, flags, exec_mask, instruction_flags, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[2][32];
  for (int lane = 0; lane < 32; ++lane) {
    double x = rocjitsu::arith::fp64_input(a, lane, instruction_flags), y = 0;
    if constexpr (rocjitsu::arith::fp64_sources(Op) >= 2) {
      y = rocjitsu::arith::fp64_input(b, lane, instruction_flags >> 1);
    }
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      x = rocjitsu::arith::fp64_flush_denorm(x);
      y = rocjitsu::arith::fp64_flush_denorm(y);
    }
    double value;
    if constexpr (Op == rocjitsu::arith::Fp64::Add || Op == rocjitsu::arith::Fp64::Mul) {
      const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
      const uint64_t xb = rocjitsu::arith::double_bits(x), yb = rocjitsu::arith::double_bits(y);
      constexpr uint64_t exponent = 0x7ff0000000000000ULL;
      if (rounding && (xb & exponent) != exponent && (yb & exponent) != exponent) {
        // Express ADD as A*1+B and MUL as A*B+signed-zero; the latter preserves
        // the product sign when both terms are zero.
        const uint64_t multiplier = Op == rocjitsu::arith::Fp64::Add ? 0x3ff0000000000000ULL : yb;
        const uint64_t addend =
            Op == rocjitsu::arith::Fp64::Add ? yb : (xb ^ yb) & 0x8000000000000000ULL;
        value = rocjitsu::arith::as_double(
            rocjitsu::arith::fma_finite_f64(xb, multiplier, addend, rounding));
      } else {
        value = Op == rocjitsu::arith::Fp64::Add ? x + y : x * y;
      }
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Fma) {
      double z = rocjitsu::arith::fp64_input(c, lane, instruction_flags >> 2);
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        z = rocjitsu::arith::fp64_flush_denorm(z);
      }
      const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
      const uint64_t xb = rocjitsu::arith::double_bits(x), yb = rocjitsu::arith::double_bits(y),
                     zb = rocjitsu::arith::double_bits(z);
      constexpr uint64_t exponent = 0x7ff0000000000000ULL;
      value =
          rounding && (xb & exponent) != exponent && (yb & exponent) != exponent &&
                  (zb & exponent) != exponent
              ? rocjitsu::arith::as_double(rocjitsu::arith::fma_finite_f64(xb, yb, zb, rounding))
              : __builtin_fma(x, y, z);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::MinLegacy ||
                  Op == rocjitsu::arith::Fp64::MaxLegacy) {
      value = rocjitsu::arith::minmax_rdna3<Op == rocjitsu::arith::Fp64::MaxLegacy>(
          x, y, flags & RJ_ARITH_FP_IEEE_MODE);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::MinNum) {
      value = rocjitsu::arith::minmax<false, false>(x, y);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::MaxNum) {
      value = rocjitsu::arith::minmax<true, false>(x, y);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Minimum) {
      value = rocjitsu::arith::minmax<false, true>(x, y);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Maximum) {
      value = rocjitsu::arith::minmax<true, true>(x, y);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::FrexpMant) {
      uint64_t magnitude = rocjitsu::arith::double_bits(x) & 0x7fffffffffffffffULL;
      int exponent;
      value =
          magnitude == 0 || magnitude >= 0x7ff0000000000000ULL ? x : __builtin_frexp(x, &exponent);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Trunc) {
      value = __builtin_trunc(x);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Ceil) {
      value = __builtin_ceil(x);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Rndne) {
      value = rocjitsu::arith::fp64_rndne(x);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Floor) {
      value = __builtin_floor(x);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Sqrt) {
      value = __builtin_sqrt(x);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Rcp) {
      value = 1.0 / x;
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Rsq) {
      value = 1.0 / __builtin_sqrt(x);
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Fract) {
      const uint64_t bits = rocjitsu::arith::double_bits(x);
      const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
      // FRACT rounds the subtraction in the guest mode, then caps its range.
      value = rounding && (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL
                  ? rocjitsu::arith::as_double(rocjitsu::arith::fma_finite_f64(
                        bits, 0x3ff0000000000000ULL,
                        rocjitsu::arith::double_bits(-__builtin_floor(x)), rounding))
                  : x - __builtin_floor(x);
      double limit = rocjitsu::arith::as_double(0x3fefffffffffffffULL);
      // GPU FRACT canonicalizes a zero fractional part, including round-down.
      if (value == 0) {
        value = 0;
      }
      if (value > limit) {
        value = limit;
      }
    }
    if constexpr (Op == rocjitsu::arith::Fp64::Trunc || Op == rocjitsu::arith::Fp64::Ceil ||
                  Op == rocjitsu::arith::Fp64::Floor || Op == rocjitsu::arith::Fp64::Rndne) {
      // Integral rounding quiets NaNs before output scaling.
      uint64_t bits = rocjitsu::arith::double_bits(value);
      if ((bits & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL) {
        value = rocjitsu::arith::as_double(bits | 0x0008000000000000ULL);
      }
    }
    const bool preserve_nan =
        (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) &&
        (rocjitsu::arith::double_bits(value) & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL;
    if constexpr (rocjitsu::arith::is_selection(Op)) {
      value = rocjitsu::arith::as_double(rocjitsu::arith::selection_output<64>(
          rocjitsu::arith::double_bits(value), instruction_flags, flags));
    } else {
      value = rocjitsu::arith::fp64_output(
          value, preserve_nan ? instruction_flags & ~RJ_ARITH_ALU_CLAMP : instruction_flags,
          unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24));
    }
    // Selection preserves a selected subnormal; arithmetic/unary outputs flush.
    constexpr bool selection = rocjitsu::arith::is_selection(Op);
    if (!selection && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
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

// Evaluate FP64 arithmetic or selection for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::Fp64 Op, bool Rdna3 = false>
int fp64_wave64(uint64_t flags, uint64_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error =
          fp64<Op, Rdna3>(flags, 0, instruction_flags, nullptr, nullptr, nullptr, nullptr)) {
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
    if (int error = fp64<Op, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), instruction_flags,
                                    dest, x, y, z)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Add the operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_add_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Add, true>(flags, exec_mask, instruction_flags, d, a, b,
                                                nullptr);
}

// Add the operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_add_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Add>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_mul_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Mul, true>(flags, exec_mask, instruction_flags, d, a, b,
                                                nullptr);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_mul_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Mul>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Fma, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Fma>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_trunc_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Trunc>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                            nullptr);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_ceil_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Ceil>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                           nullptr);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_rndne_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Rndne>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                            nullptr);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_floor_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Floor>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                            nullptr);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fract_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Fract>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                            nullptr);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_sqrt_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Sqrt>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                           nullptr);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_rcp_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Rcp>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                          nullptr);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_rsq_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Rsq>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                          nullptr);
}

// Select the minimum with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_min_num_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::MinNum>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Select the maximum with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_max_num_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::MaxNum>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Select the NaN-propagating minimum for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_minimum_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Minimum>(flags, exec_mask, instruction_flags, d, a, b,
                                              nullptr);
}

// Select the NaN-propagating maximum for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_maximum_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Maximum>(flags, exec_mask, instruction_flags, d, a, b,
                                              nullptr);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_frexp_mant_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a,
                                        [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::FrexpMant>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                                nullptr);
}

// Select the minimum for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_min_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::MinLegacy, true>(flags, exec_mask, instruction_flags, d, a, b,
                                                      nullptr);
}

// Select the maximum for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_max_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::MaxLegacy, true>(flags, exec_mask, instruction_flags, d, a, b,
                                                      nullptr);
}

// Add the operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_add_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Add, true>(flags, exec_mask, instruction_flags, d, a, b,
                                                       nullptr);
}

// Add the operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_add_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Add>(flags, exec_mask, instruction_flags, d, a, b,
                                                 nullptr);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_mul_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Mul, true>(flags, exec_mask, instruction_flags, d, a, b,
                                                       nullptr);
}

// Multiply the operands for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_mul_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Mul>(flags, exec_mask, instruction_flags, d, a, b,
                                                 nullptr);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Fma, true>(flags, exec_mask, instruction_flags, d, a, b,
                                                       c);
}

// Compute fused A*B+C for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fma_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Fma>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_trunc_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Trunc>(flags, exec_mask, instruction_flags, d, a,
                                                   nullptr, nullptr);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_ceil_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Ceil>(flags, exec_mask, instruction_flags, d, a,
                                                  nullptr, nullptr);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_rndne_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Rndne>(flags, exec_mask, instruction_flags, d, a,
                                                   nullptr, nullptr);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_floor_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Floor>(flags, exec_mask, instruction_flags, d, a,
                                                   nullptr, nullptr);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fract_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Fract>(flags, exec_mask, instruction_flags, d, a,
                                                   nullptr, nullptr);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_sqrt_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Sqrt>(flags, exec_mask, instruction_flags, d, a,
                                                  nullptr, nullptr);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_rcp_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Rcp>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                                 nullptr);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_rsq_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Rsq>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                                 nullptr);
}

// Select the minimum with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_min_num_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::MinNum>(flags, exec_mask, instruction_flags, d, a, b,
                                                    nullptr);
}

// Select the maximum with numeric-preferred NaN selection for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_max_num_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::MaxNum>(flags, exec_mask, instruction_flags, d, a, b,
                                                    nullptr);
}

// Select the NaN-propagating minimum for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_minimum_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Minimum>(flags, exec_mask, instruction_flags, d, a, b,
                                                     nullptr);
}

// Select the NaN-propagating maximum for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_maximum_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b,
                               [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Maximum>(flags, exec_mask, instruction_flags, d, a, b,
                                                     nullptr);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_frexp_mant_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a,
                                        [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::FrexpMant>(flags, exec_mask, instruction_flags, d, a,
                                                       nullptr, nullptr);
}

// Select the minimum for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_min_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::MinLegacy, true>(flags, exec_mask, instruction_flags, d,
                                                             a, b, nullptr);
}

// Select the maximum for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_max_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::MaxLegacy, true>(flags, exec_mask, instruction_flags, d,
                                                             a, b, nullptr);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_trunc_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Trunc, true>(flags, exec_mask, instruction_flags, d, a,
                                                  nullptr, nullptr);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_trunc_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Trunc, true>(flags, exec_mask, instruction_flags, d, a,
                                                         nullptr, nullptr);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_ceil_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Ceil, true>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                                 nullptr);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_ceil_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Ceil, true>(flags, exec_mask, instruction_flags, d, a,
                                                        nullptr, nullptr);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_rndne_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Rndne, true>(flags, exec_mask, instruction_flags, d, a,
                                                  nullptr, nullptr);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_rndne_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Rndne, true>(flags, exec_mask, instruction_flags, d, a,
                                                         nullptr, nullptr);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_floor_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Floor, true>(flags, exec_mask, instruction_flags, d, a,
                                                  nullptr, nullptr);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_floor_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Floor, true>(flags, exec_mask, instruction_flags, d, a,
                                                         nullptr, nullptr);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fract_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Fract, true>(flags, exec_mask, instruction_flags, d, a,
                                                  nullptr, nullptr);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_fract_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Fract, true>(flags, exec_mask, instruction_flags, d, a,
                                                         nullptr, nullptr);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_sqrt_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Sqrt, true>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                                 nullptr);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_sqrt_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Sqrt, true>(flags, exec_mask, instruction_flags, d, a,
                                                        nullptr, nullptr);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_rcp_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Rcp, true>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                                nullptr);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_rcp_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Rcp, true>(flags, exec_mask, instruction_flags, d, a,
                                                       nullptr, nullptr);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_rsq_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::Rsq, true>(flags, exec_mask, instruction_flags, d, a, nullptr,
                                                nullptr);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_rsq_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::Rsq, true>(flags, exec_mask, instruction_flags, d, a,
                                                       nullptr, nullptr);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_frexp_mant_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64<rocjitsu::arith::Fp64::FrexpMant, true>(flags, exec_mask, instruction_flags, d, a,
                                                      nullptr, nullptr);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs; required FP-status reporting is unsupported.
int rj_arith_v_frexp_mant_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return fp64_wave64<rocjitsu::arith::Fp64::FrexpMant, true>(flags, exec_mask, instruction_flags, d,
                                                             a, nullptr, nullptr);
}
