// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 unary math, integral rounding, and special-value policies.
// Owns public entry points and their validation/dispatch policy.

#include "math/half_unary.h"
#include "common/alu.h"
#include "common/float_formats.h"
#include "common/internal.h"
#include "math/half_unary_exact.h"
#include "math/unary_impl.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

template <rocjitsu::arith::Unary Op>
constexpr bool supports_exact =
    Op == rocjitsu::arith::Unary::Trunc || Op == rocjitsu::arith::Unary::Ceil ||
    Op == rocjitsu::arith::Unary::Rndne || Op == rocjitsu::arith::Unary::Floor ||
    Op == rocjitsu::arith::Unary::FrexpMant || Op == rocjitsu::arith::Unary::Exp ||
    Op == rocjitsu::arith::Unary::Log || rocjitsu::arith::half_unary_sfu<Op>;

static const uint64_t fp_flags = RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                 RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE |
                                 RJ_ARITH_FP_DX10_CLAMP_OFF;

// Evaluate FP16 unary math for the selected lanes. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::Unary Op, bool Rdna3 = false, bool PreserveClampNan = false>
[[gnu::noinline]] int unary_exact(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(flags, exec_mask, mode, a,
                                        [&](uint32_t exec_mask, const uint32_t *const *source) {
                                          return unary_exact<Op, Rdna3, PreserveClampNan>(
                                              flags, exec_mask, uint32_t(mode), d, source);
                                        });
  }
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_MASK |
                         RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, fp_flags)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if constexpr (rocjitsu::arith::half_unary_sfu<Op>) {
    if (!(flags & RJ_ARITH_FP_ROUND_MASK) &&
        rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
      rocjitsu::arith::half_sfu_exact_x86_64_v3<Op>(flags, exec_mask, uint32_t(mode), d[0], a[0],
                                                    !Rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE));
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
  if constexpr (Op == rocjitsu::arith::Unary::Trunc || Op == rocjitsu::arith::Unary::Ceil ||
                Op == rocjitsu::arith::Unary::Floor || Op == rocjitsu::arith::Unary::Rndne) {
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if (!(flags & RJ_ARITH_FP_ROUND_MASK) &&
        rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
      rocjitsu::arith::half_round_exact_x86_64_v4<Op>(flags, exec_mask, uint32_t(mode), d[0], a[0]);
      return RJ_ARITH_SUCCESS;
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if (!(flags & RJ_ARITH_FP_ROUND_MASK) &&
        rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
      rocjitsu::arith::half_unary_x86_64_v3<Op>(flags, exec_mask, uint32_t(mode), d[0], a[0]);
      return RJ_ARITH_SUCCESS;
    }
#endif
  }
  unsigned a_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  unsigned d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    result[lane] = rocjitsu::arith::half_unary_exact_value<Op, Rdna3, PreserveClampNan>(
        flags, uint32_t(mode), uint16_t(a[0][lane] >> a_shift));
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(65535U << d_shift)) | (uint32_t(result[lane]) << d_shift);
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate FP16 unary math for the selected lanes. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::Unary Op, bool GuestDenormals = true, bool PreserveClampNan = false>
[[gnu::noinline]] int unary(uint64_t flags, uint32_t exec_mask, uint64_t modifiers,
                            uint32_t *const *d, const uint32_t *const *a) {
  if (modifiers >> 32) {
    return rocjitsu::arith::execute_dpp(flags, exec_mask, modifiers, a,
                                        [&](uint32_t exec_mask, const uint32_t *const *source) {
                                          return unary<Op, GuestDenormals, PreserveClampNan>(
                                              flags, exec_mask, uint32_t(modifiers), d, source);
                                        });
  }
  const uint32_t known = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_MASK |
                         RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D;
  if (int error = rocjitsu::arith::validate(flags, modifiers & ~known, fp_flags)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (!(flags & RJ_ARITH_FP_ROUND_MASK) &&
      rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::half_unary_x86_64_v3<Op, GuestDenormals>(flags, exec_mask, modifiers, d[0],
                                                              a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  if constexpr (supports_exact<Op> && !rocjitsu::arith::half_unary_sfu<Op> &&
                Op != rocjitsu::arith::Unary::Exp && Op != rocjitsu::arith::Unary::Log) {
    if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
      return unary_exact<Op, false, true>(flags, exec_mask, modifiers, d, a);
    }
    return unary_exact<Op>(flags, exec_mask, modifiers, d, a);
  }
  int a_shift = modifiers & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  int d_shift = modifiers & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint16_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint16_t input = uint16_t(a[0][lane] >> a_shift);
    if constexpr (rocjitsu::arith::half_unary_sfu<Op> || Op == rocjitsu::arith::Unary::Fract ||
                  Op == rocjitsu::arith::Unary::Exp || Op == rocjitsu::arith::Unary::Log) {
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) &&
          (input & 0x7fff) < 0x400) {
        input &= 0x8000;
      }
    }
    float x = rocjitsu::arith::alu_input(
        rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(input)), modifiers);
    if constexpr (Op == rocjitsu::arith::Unary::Exp) {
      // Bound finite intermediates without changing any final FP16 result,
      // including finite-overflow saturation and output scaling by up to four.
      if (__builtin_isfinite(x)) {
        x = rocjitsu::arith::clamp(x, -64.0f, 64.0f);
      }
    }
    float value = rocjitsu::arith::unary_value<Op>(x);
    if constexpr (Op == rocjitsu::arith::Unary::Log) {
      if (x == 0 && (flags & RJ_ARITH_FP16_OVFL)) {
        value = -65504;
      }
    }
    if constexpr (Op == rocjitsu::arith::Unary::Rcp || Op == rocjitsu::arith::Unary::Rsq) {
      if ((flags & RJ_ARITH_FP16_OVFL) && x == 0) {
        value = __builtin_copysignf(65504.0f, value);
      }
    }
    if constexpr (Op == rocjitsu::arith::Unary::Exp || Op == rocjitsu::arith::Unary::Log ||
                  rocjitsu::arith::half_unary_sfu<Op>) {
      // Power-of-two scaling commutes with normal half rounding. For these
      // three SFUs, CLAMP also removes the overflow distinction, so loose mode
      // can omit the intermediate conversion when CLAMP is present.
      if ((modifiers & RJ_ARITH_ALU_OMOD_MASK) &&
          (!rocjitsu::arith::half_unary_sfu<Op> || !(modifiers & RJ_ARITH_ALU_CLAMP))) {
        value = rocjitsu::arith::f16_to_float(
            rocjitsu::arith::float_to_f16(value, flags & RJ_ARITH_FP16_OVFL));
      }
    }
    if constexpr (Op == rocjitsu::arith::Unary::Exp || Op == rocjitsu::arith::Unary::Log) {
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
        uint16_t rounded = rocjitsu::arith::float_to_f16(value, flags & RJ_ARITH_FP16_OVFL);
        if ((rounded & 0x7fff) < 0x400) {
          rounded &= 0x8000;
        }
        value = rocjitsu::arith::f16_to_float(rounded);
      }
    }
    if constexpr (Op == rocjitsu::arith::Unary::Fract) {
      // Round the exact half-input subtraction before OMOD, using guest MODE.
      if (__builtin_isfinite(x)) {
        uint16_t half = rocjitsu::arith::half_fma_narrow(
            rocjitsu::arith::bit_cast<uint64_t>(double(x) - __builtin_floor(double(x))),
            flags & RJ_ARITH_FP16_OVFL, unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24));
        value = rocjitsu::arith::f16_to_float(half);
      }
      // Keep a fraction strictly below one after FP16 rounding.
      if (value > 0x1.ffcp-1f) {
        value = 0x1.ffcp-1f;
      }
    }
    if constexpr (rocjitsu::arith::half_unary_sfu<Op> || Op == rocjitsu::arith::Unary::Fract) {
      if (GuestDenormals && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
          __builtin_fabs(value) < 0x1p-14f) {
        value = __builtin_copysignf(0.0f, value);
      }
    }
    uint32_t output_mode = modifiers;
    if constexpr (PreserveClampNan) {
      if ((rocjitsu::arith::as_bits(value) & 0x7fffffff) > 0x7f800000) {
        output_mode &= ~RJ_ARITH_ALU_CLAMP;
      }
    }
    float output = rocjitsu::arith::alu_output_f16(value, output_mode);
    result[lane] =
        (flags & RJ_ARITH_FP_ROUND_MASK) && __builtin_isfinite(output)
            ? rocjitsu::arith::half_fma_narrow(rocjitsu::arith::bit_cast<uint64_t>(double(output)),
                                               flags & RJ_ARITH_FP16_OVFL,
                                               unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24))
            : rocjitsu::arith::float_to_f16(output, flags & RJ_ARITH_FP16_OVFL);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(0xffffU << d_shift)) | (uint32_t(result[lane]) << d_shift);
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate FP16 unary math for the selected lanes. Requires validated controls; preserve
// inactive destination lanes.
template <rocjitsu::arith::Unary Op, bool Rdna3 = false>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, [[maybe_unused]] uint32_t *state) {
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if constexpr (!Rdna3) {
    if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
      return RJ_ARITH_ERROR_INVALID_FLAGS;
    }
  }
  if constexpr (Rdna3) {
    if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
      mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
    }
  }
  if constexpr (supports_exact<Op>) {
    if (rocjitsu::arith::is_exact(flags)) {
      constexpr bool rdna3_nan_policy =
          Rdna3 && (rocjitsu::arith::half_unary_sfu<Op> || Op == rocjitsu::arith::Unary::Exp ||
                    Op == rocjitsu::arith::Unary::Log);
      if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
        return unary_exact<Op, rdna3_nan_policy, true>(flags, exec_mask, mode, d, a);
      }
      return unary_exact<Op, rdna3_nan_policy>(flags, exec_mask, mode, d, a);
    }
  }
  if constexpr (Op == rocjitsu::arith::Unary::Exp || Op == rocjitsu::arith::Unary::Log) {
    if (!(flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) {
      if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
        return unary<Op, false, true>(flags, exec_mask, mode, d, a);
      }
      return unary<Op, false>(flags, exec_mask, mode, d, a);
    }
  }
  if constexpr (!supports_exact<Op> || rocjitsu::arith::half_unary_sfu<Op> ||
                Op == rocjitsu::arith::Unary::Exp || Op == rocjitsu::arith::Unary::Log) {
    if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
      return unary<Op, true, true>(flags, exec_mask, mode, d, a);
    }
  }
  return unary<Op>(flags, exec_mask, mode, d, a);
}

// Evaluate FP16 unary math for Wave64. Reject invalid controls before changing outputs.
template <rocjitsu::arith::Unary Op, bool Rdna3 = false>
int unary_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, [[maybe_unused]] uint32_t *state) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<Op, Rdna3>(flags, 0, mode, nullptr, nullptr, state)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_instruction_flags = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_instruction_flags =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    if (int error = run<Op, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)),
                                   half_instruction_flags, &dest, &source, state)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Trunc, true>(flags, exec_mask, instruction_flags, d, a,
                                                  trapsts);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Trunc, false>(flags, exec_mask, instruction_flags, d, a,
                                                   excp_flag_user);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Ceil, true>(flags, exec_mask, instruction_flags, d, a,
                                                 trapsts);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Ceil, false>(flags, exec_mask, instruction_flags, d, a,
                                                  excp_flag_user);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Rndne, true>(flags, exec_mask, instruction_flags, d, a,
                                                  trapsts);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Rndne, false>(flags, exec_mask, instruction_flags, d, a,
                                                   excp_flag_user);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Floor, true>(flags, exec_mask, instruction_flags, d, a,
                                                  trapsts);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Floor, false>(flags, exec_mask, instruction_flags, d, a,
                                                   excp_flag_user);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Sqrt, true>(flags, exec_mask, instruction_flags, d, a,
                                                 trapsts);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Sqrt, false>(flags, exec_mask, instruction_flags, d, a,
                                                  excp_flag_user);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Rcp, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Rcp, false>(flags, exec_mask, instruction_flags, d, a,
                                                 excp_flag_user);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Rsq, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Rsq, false>(flags, exec_mask, instruction_flags, d, a,
                                                 excp_flag_user);
}

// Compute base-two exponentiation for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_exp_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Exp, false>(flags, exec_mask, instruction_flags, d, a,
                                                 excp_flag_user);
}

// Compute the base-two logarithm for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_log_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Log, false>(flags, exec_mask, instruction_flags, d, a,
                                                 excp_flag_user);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Fract, true>(flags, exec_mask, instruction_flags, d, a,
                                                  trapsts);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Fract, false>(flags, exec_mask, instruction_flags, d, a,
                                                   excp_flag_user);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_mant_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::FrexpMant, true>(flags, exec_mask, instruction_flags, d, a,
                                                      trapsts);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_mant_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::FrexpMant, false>(flags, exec_mask, instruction_flags, d, a,
                                                       excp_flag_user);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Trunc, true>(flags, exec_mask, instruction_flags, d,
                                                           a, trapsts);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Trunc, false>(flags, exec_mask, instruction_flags, d,
                                                            a, excp_flag_user);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Ceil, true>(flags, exec_mask, instruction_flags, d, a,
                                                          trapsts);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Ceil, false>(flags, exec_mask, instruction_flags, d,
                                                           a, excp_flag_user);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rndne, true>(flags, exec_mask, instruction_flags, d,
                                                           a, trapsts);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rndne, false>(flags, exec_mask, instruction_flags, d,
                                                            a, excp_flag_user);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Floor, true>(flags, exec_mask, instruction_flags, d,
                                                           a, trapsts);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Floor, false>(flags, exec_mask, instruction_flags, d,
                                                            a, excp_flag_user);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Sqrt, true>(flags, exec_mask, instruction_flags, d, a,
                                                          trapsts);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Sqrt, false>(flags, exec_mask, instruction_flags, d,
                                                           a, excp_flag_user);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rcp, true>(flags, exec_mask, instruction_flags, d, a,
                                                         trapsts);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rcp, false>(flags, exec_mask, instruction_flags, d, a,
                                                          excp_flag_user);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rsq, true>(flags, exec_mask, instruction_flags, d, a,
                                                         trapsts);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rsq, false>(flags, exec_mask, instruction_flags, d, a,
                                                          excp_flag_user);
}

// Compute base-two exponentiation for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_exp_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Exp, false>(flags, exec_mask, instruction_flags, d, a,
                                                          excp_flag_user);
}

// Compute the base-two logarithm for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_log_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Log, false>(flags, exec_mask, instruction_flags, d, a,
                                                          excp_flag_user);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Fract, true>(flags, exec_mask, instruction_flags, d,
                                                           a, trapsts);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Fract, false>(flags, exec_mask, instruction_flags, d,
                                                            a, excp_flag_user);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_mant_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::FrexpMant, true>(flags, exec_mask, instruction_flags,
                                                               d, a, trapsts);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_mant_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::FrexpMant, false>(flags, exec_mask, instruction_flags,
                                                                d, a, excp_flag_user);
}

// Compute base-two exponentiation for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_exp_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Exp, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Compute the base-two logarithm for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_log_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<rocjitsu::arith::Unary::Log, true>(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Compute base-two exponentiation for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_exp_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Exp, true>(flags, exec_mask, instruction_flags, d, a,
                                                         trapsts);
}

// Compute the base-two logarithm for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_log_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Log, true>(flags, exec_mask, instruction_flags, d, a,
                                                         trapsts);
}
