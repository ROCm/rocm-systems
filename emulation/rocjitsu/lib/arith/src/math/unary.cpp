// SPDX-License-Identifier: MIT

// FP32 unary math, integral rounding, and mantissa decomposition.
// Owns public entry points and their validation/dispatch policy.
// FP32 transcendental flushing follows rocjitsu's shared execution models
// and is checked against gfx1201 hardware captures.

#include "math/unary.h"
#include "common/alu.h"
#include "common/internal.h"
#include "math/unary_exact.h"
#include "math/x86_64/rcp_exact.h"
#include "math/x86_64/root_exact.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Exact unary mappings with the inherited scalar models as fallback.
template <rocjitsu::arith::Unary Op>
[[gnu::noinline]] int unary_exact(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a, bool rdna3,
                                  uint32_t *state) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return unary_exact<Op>(flags, exec_mask, uint32_t(mode), d, source, rdna3, state);
        });
  }
  const uint32_t known =
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~known, true,
          RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
              RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | (rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  if (rdna3 &&
      ((flags & RJ_ARITH_FP_IEEE_MODE) || (!rocjitsu::arith::unary_flushes_f32<Op> &&
                                           !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)))) {
    mode &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  if constexpr (Op == rocjitsu::arith::Unary::Rcp) {
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
      rocjitsu::arith::rcp_exact_x86_64_v4(flags, exec_mask, uint32_t(mode), d[0], a[0],
                                           !rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE));
      return RJ_ARITH_SUCCESS;
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
      rocjitsu::arith::rcp_exact_x86_64_v3(flags, exec_mask, uint32_t(mode), d[0], a[0],
                                           !rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE));
      return RJ_ARITH_SUCCESS;
    }
#endif
  }
  if constexpr (Op == rocjitsu::arith::Unary::Rsq || Op == rocjitsu::arith::Unary::Sqrt) {
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
      uint32_t causes = rocjitsu::arith::root_exact_x86_64_v4(
          Op == rocjitsu::arith::Unary::Rsq, flags, exec_mask, uint32_t(mode), d[0], a[0],
          !rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE), rocjitsu::arith::reports_fp_status(flags));
      if (rocjitsu::arith::reports_fp_status(flags)) {
        *state |= causes;
      }
      return RJ_ARITH_SUCCESS;
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
      uint32_t causes = rocjitsu::arith::root_exact_x86_64_v3(
          Op == rocjitsu::arith::Unary::Rsq, flags, exec_mask, uint32_t(mode), d[0], a[0],
          !rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE), rocjitsu::arith::reports_fp_status(flags));
      if (rocjitsu::arith::reports_fp_status(flags)) {
        *state |= causes;
      }
      return RJ_ARITH_SUCCESS;
    }
#endif
  }
  if constexpr (rocjitsu::arith::unary_rounds_f32<Op>) {
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
      rocjitsu::arith::round_exact_x86_64_v4(Op, exec_mask, uint32_t(mode), d[0], a[0], flags);
      return RJ_ARITH_SUCCESS;
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
      rocjitsu::arith::round_exact_x86_64_v3(Op, exec_mask, uint32_t(mode), d[0], a[0], flags);
      return RJ_ARITH_SUCCESS;
    }
#endif
  }
  uint32_t causes = 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    if constexpr (Op == rocjitsu::arith::Unary::Sqrt) {
      // Borrow rocjitsu's classify_sqrt_f32_vop3: test the modified input
      // before flushing or CLAMP. Negative zero and NaNs do not contribute.
      if (rocjitsu::arith::reports_fp_status(flags) && ((exec_mask >> lane) & 1)) {
        uint32_t input = a[0][lane];
        if (mode & RJ_ARITH_ALU_ABS_A) {
          input &= 0x7fffffffU;
        }
        if (mode & RJ_ARITH_ALU_NEG_A) {
          input ^= 0x80000000U;
        }
        const uint32_t magnitude = input & 0x7fffffffU;
        if ((input >> 31) && magnitude != 0 && magnitude <= 0x7f800000U) {
          causes |= 1U;
        }
      }
    }
    result[lane] = rocjitsu::arith::unary_exact_value<Op>(flags, uint32_t(mode), a[0][lane], rdna3);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    *state |= causes;
  }
  return RJ_ARITH_SUCCESS;
}

// Separate the two scalar clamp policies at compile time so the default loop
// retains its existing vectorization. Both architectures share these kernels.
template <rocjitsu::arith::Unary Op, bool PreserveClampNan = false>
int unary_loose(uint64_t flags, uint32_t exec_mask, uint64_t modifiers, uint32_t *const *d,
                const uint32_t *const *a) {
  if (modifiers >> 32) {
    return rocjitsu::arith::execute_dpp(flags, exec_mask, modifiers, a,
                                        [&](uint32_t exec_mask, const uint32_t *const *source) {
                                          return unary_loose<Op, PreserveClampNan>(
                                              flags, exec_mask, uint32_t(modifiers), d, source);
                                        });
  }

  const uint32_t known =
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP;
  constexpr uint64_t fp_flags = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE |
                                (PreserveClampNan ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0);
  if (int error = rocjitsu::arith::validate(flags, modifiers & ~known, false, fp_flags)) {
    return error;
  }
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::unary_x86_64_v3(Op, exec_mask, modifiers, d[0], a[0], flags);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float value = rocjitsu::arith::alu_input(a[0][lane], modifiers);

    // Some special functions flush unconditionally; other operations follow the guest denormal
    // controls.
    if constexpr (rocjitsu::arith::unary_flushes_f32<Op>) {
      value = rocjitsu::arith::flush_denorm_f32(value);
    } else {
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        value = rocjitsu::arith::flush_denorm_f32(value);
      }
    }
    value = rocjitsu::arith::unary_value<Op>(value);
    if constexpr (rocjitsu::arith::unary_flushes_f32<Op>) {
      value = rocjitsu::arith::flush_denorm_f32(value);
    } else if constexpr (Op == rocjitsu::arith::Unary::Fract) {
      if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && !(modifiers & RJ_ARITH_ALU_OMOD_HALF)) {
        value = rocjitsu::arith::flush_denorm_f32(value);
      }
    }
    uint32_t output_mode = modifiers;
    if constexpr (PreserveClampNan) {
      if ((rocjitsu::arith::as_bits(value) & 0x7fffffff) > 0x7f800000) {
        output_mode &= ~RJ_ARITH_ALU_CLAMP;
      }
    }
    result[lane] = rocjitsu::arith::as_bits(rocjitsu::arith::alu_output_f32(value, output_mode));
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Evaluate FP32 unary math for the selected lanes. Requires validated controls; preserve
// inactive destination lanes.
template <rocjitsu::arith::Unary Op, bool Rdna3 = false>
int unary(uint64_t flags, uint32_t exec_mask, uint64_t modifiers, uint32_t *const *d,
          const uint32_t *const *a, uint32_t *excp_flag_user = nullptr) {
  if (rocjitsu::arith::reports_fp_status(flags) &&
      (Op != rocjitsu::arith::Unary::Sqrt || !rocjitsu::arith::is_exact(flags))) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if constexpr (Op != rocjitsu::arith::Unary::Fract) {
    if (rocjitsu::arith::is_exact(flags)) {
      return unary_exact<Op>(flags, exec_mask, modifiers, d, a, Rdna3, excp_flag_user);
    }
  }
  if constexpr (Rdna3) {
    // Transcendentals use forced-flush OMOD policy. Other unary operations
    // use the guest output-denormal policy, even where their result cannot
    // be subnormal. Both policies also honor RDNA3 IEEE mode.
    if ((flags & RJ_ARITH_FP_IEEE_MODE) || (!rocjitsu::arith::unary_flushes_f32<Op> &&
                                            !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) {
      modifiers &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
    }
    if (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) {
      return unary_loose<Op, true>(flags, exec_mask, modifiers, d, a);
    }
  }
  return unary_loose<Op>(flags, exec_mask, modifiers, d, a);
}

// Evaluate FP32 unary math for Wave64. Reject invalid controls before changing outputs.
template <rocjitsu::arith::Unary Op, bool Rdna3 = false>
int unary_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                 const uint32_t *const *a, uint32_t *excp_flag_user = nullptr) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = unary<Op, Rdna3>(flags, 0, mode, nullptr, nullptr, excp_flag_user)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_mode =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source = a[0] + half * 32;
    if (int error = unary<Op, Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                     &source, excp_flag_user)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Trunc, true>(flags, exec_mask, instruction_flags, d, a);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Trunc>(flags, exec_mask, instruction_flags, d, a,
                                              excp_flag_user);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Ceil, true>(flags, exec_mask, instruction_flags, d, a);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Ceil>(flags, exec_mask, instruction_flags, d, a,
                                             excp_flag_user);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Rndne, true>(flags, exec_mask, instruction_flags, d, a);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Rndne>(flags, exec_mask, instruction_flags, d, a,
                                              excp_flag_user);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Floor, true>(flags, exec_mask, instruction_flags, d, a);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Floor>(flags, exec_mask, instruction_flags, d, a,
                                              excp_flag_user);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary<rocjitsu::arith::Unary::Sqrt, true>(flags, exec_mask, instruction_flags, d, a,
                                                   trapsts);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary<rocjitsu::arith::Unary::Sqrt>(flags, exec_mask, instruction_flags, d, a,
                                             excp_flag_user);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Rcp, true>(flags, exec_mask, instruction_flags, d, a);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Rcp>(flags, exec_mask, instruction_flags, d, a,
                                            excp_flag_user);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Rsq, true>(flags, exec_mask, instruction_flags, d, a);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Rsq>(flags, exec_mask, instruction_flags, d, a,
                                            excp_flag_user);
}

// Compute base-two exponentiation for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_exp_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Exp, true>(flags, exec_mask, instruction_flags, d, a);
}

// Compute base-two exponentiation for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_exp_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Exp>(flags, exec_mask, instruction_flags, d, a,
                                            excp_flag_user);
}

// Compute the base-two logarithm for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_log_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Log, true>(flags, exec_mask, instruction_flags, d, a);
}

// Compute the base-two logarithm for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_log_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Log>(flags, exec_mask, instruction_flags, d, a,
                                            excp_flag_user);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Fract, true>(flags, exec_mask, instruction_flags, d, a);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::Fract>(flags, exec_mask, instruction_flags, d, a,
                                              excp_flag_user);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_mant_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::FrexpMant, true>(flags, exec_mask, mode, d, a);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_mant_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary<rocjitsu::arith::Unary::FrexpMant>(flags, exec_mask, mode, d, a, excp_flag_user);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Trunc, true>(flags, exec_mask, instruction_flags, d,
                                                           a);
}

// Round toward zero for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_trunc_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Trunc>(flags, exec_mask, instruction_flags, d, a,
                                                     excp_flag_user);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a,
                            [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Ceil, true>(flags, exec_mask, instruction_flags, d,
                                                          a);
}

// Round toward positive infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_ceil_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Ceil>(flags, exec_mask, instruction_flags, d, a,
                                                    excp_flag_user);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rndne, true>(flags, exec_mask, instruction_flags, d,
                                                           a);
}

// Round to the nearest integer with ties to even for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rndne_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rndne>(flags, exec_mask, instruction_flags, d, a,
                                                     excp_flag_user);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Floor, true>(flags, exec_mask, instruction_flags, d,
                                                           a);
}

// Round toward negative infinity for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_floor_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Floor>(flags, exec_mask, instruction_flags, d, a,
                                                     excp_flag_user);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Sqrt, true>(flags, exec_mask, instruction_flags, d, a,
                                                          trapsts);
}

// Compute the square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_sqrt_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return unary_wave64<rocjitsu::arith::Unary::Sqrt>(flags, exec_mask, instruction_flags, d, a,
                                                    excp_flag_user);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rcp, true>(flags, exec_mask, instruction_flags, d, a);
}

// Compute the reciprocal for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rcp_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rcp>(flags, exec_mask, instruction_flags, d, a,
                                                   excp_flag_user);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rsq, true>(flags, exec_mask, instruction_flags, d, a);
}

// Compute the reciprocal square root for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_rsq_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Rsq>(flags, exec_mask, instruction_flags, d, a,
                                                   excp_flag_user);
}

// Compute base-two exponentiation for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_exp_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Exp, true>(flags, exec_mask, instruction_flags, d, a);
}

// Compute base-two exponentiation for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_exp_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Exp>(flags, exec_mask, instruction_flags, d, a,
                                                   excp_flag_user);
}

// Compute the base-two logarithm for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_log_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a,
                           [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Log, true>(flags, exec_mask, instruction_flags, d, a);
}

// Compute the base-two logarithm for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_log_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Log>(flags, exec_mask, instruction_flags, d, a,
                                                   excp_flag_user);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a,
                             [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Fract, true>(flags, exec_mask, instruction_flags, d,
                                                           a);
}

// Compute the fractional part for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_fract_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::Fract>(flags, exec_mask, instruction_flags, d, a,
                                                     excp_flag_user);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_mant_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::FrexpMant, true>(flags, exec_mask, mode, d, a);
}

// Extract the requested exponent or mantissa component for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_frexp_mant_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return unary_wave64<rocjitsu::arith::Unary::FrexpMant>(flags, exec_mask, mode, d, a,
                                                         excp_flag_user);
}
