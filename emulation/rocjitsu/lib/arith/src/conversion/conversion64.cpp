// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Conversions between FP64 and FP32 or 32-bit integers in split VGPR layouts.
// Owns public entry points and their validation/dispatch policy.

#include "conversion/conversion64.h"
#include "common/alu.h"
#include "common/internal.h"
#include "conversion/float_conversion.h"
#include "fp/fp64_impl.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Convert FP64, FP32, or 32-bit integers for the selected lanes. Reject invalid controls
// before changing outputs.
template <rocjitsu::arith::Conversion64 Op>
int convert(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
            const uint32_t *const *a, bool quiet_nan = true) {
  constexpr bool from_integer = Op == rocjitsu::arith::Conversion64::SignedToDouble ||
                                Op == rocjitsu::arith::Conversion64::UnsignedToDouble;
  constexpr bool to_double = from_integer || Op == rocjitsu::arith::Conversion64::FloatToDouble;
  const uint32_t known = RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP |
                         (from_integer ? 0 : RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_ABS_A);
  if (int error = rocjitsu::arith::validate(
          flags, mode & ~known,
          RJ_ARITH_FP_DX10_CLAMP_OFF | RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_IEEE_MODE |
              RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::conversion64_x86_64_v4<Op>(flags, exec_mask, mode, d, a, quiet_nan);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::conversion64_x86_64_v3<Op>(flags, exec_mask, mode, d, a, quiet_nan);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[to_double ? 2 : 1][32];
  for (int lane = 0; lane < 32; ++lane) {
    if constexpr (to_double || Op == rocjitsu::arith::Conversion64::DoubleToFloat) {
      uint64_t raw = a[0][lane];
      if constexpr (!to_double) {
        raw |= uint64_t(a[1][lane]) << 32;
      }
      uint64_t bits =
          rocjitsu::arith::conversion_detail::convert64<Op>(raw, flags, mode, quiet_nan);
      result[0][lane] = uint32_t(bits);
      if constexpr (to_double) {
        result[1][lane] = uint32_t(bits >> 32);
      }
    } else {
      double value = rocjitsu::arith::fp64_input(a, lane, mode);
      if constexpr (Op == rocjitsu::arith::Conversion64::DoubleToUnsigned) {
        // Match rocjitsu's sema_lower.py: truncate, saturate, map NaN to zero.
        // Classify before casting, to avoid undefined out-of-range casts.
        result[0][lane] = !(value > 0) ? 0 : value >= 4294967296.0 ? UINT32_MAX : uint32_t(value);
      } else {
        result[0][lane] = __builtin_isnan(value)   ? 0
                          : value >= 2147483648.0  ? uint32_t(INT32_MAX)
                          : value <= -2147483648.0 ? uint32_t(INT32_MIN)
                                                   : uint32_t(int32_t(value));
      }
      // Integer outputs ignore CLAMP and OMOD numerically. GPU exception
      // reporting is not part of the host FP environment contract.
    }
  }
  for (int reg = 0; reg < (to_double ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[reg][lane] = result[reg][lane];
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Convert FP64, FP32, or 32-bit integers for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::Conversion64 Op>
int convert_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
                   const uint32_t *const *a, bool quiet_nan = true) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = convert<Op>(flags, 0, mode, nullptr, nullptr, quiet_nan)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  constexpr bool to_double = Op == rocjitsu::arith::Conversion64::SignedToDouble ||
                             Op == rocjitsu::arith::Conversion64::UnsignedToDouble ||
                             Op == rocjitsu::arith::Conversion64::FloatToDouble;
  constexpr int inputs = to_double ? 1 : 2, outputs = to_double ? 2 : 1;
  for (int half = 0; half < 2; ++half) {
    uint32_t *dest[outputs];
    const uint32_t *source[inputs];
    for (int reg = 0; reg < outputs; ++reg) {
      dest[reg] = d[reg] + half * 32;
    }
    for (int reg = 0; reg < inputs; ++reg) {
      source[reg] = a[reg] + half * 32;
    }
    if (int error =
            convert<Op>(flags, uint32_t(exec_mask >> (half * 32)), mode, dest, source, quiet_nan)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Convert I32 to F64 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f64_i32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a) {
  flags &= ~RJ_ARITH_FP_DX10_CLAMP_OFF;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return convert<rocjitsu::arith::Conversion64::SignedToDouble>(flags, exec_mask, instruction_flags,
                                                                d, a);
}

// Convert U32 to F64 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f64_u32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a) {
  flags &= ~RJ_ARITH_FP_DX10_CLAMP_OFF;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return convert<rocjitsu::arith::Conversion64::UnsignedToDouble>(flags, exec_mask,
                                                                  instruction_flags, d, a);
}

// Convert F64 to I32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_i32_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags &= ~RJ_ARITH_FP_DX10_CLAMP_OFF;
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
  return convert<rocjitsu::arith::Conversion64::DoubleToSigned>(flags, exec_mask, instruction_flags,
                                                                d, a);
}

// Convert F64 to U32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_u32_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags &= ~RJ_ARITH_FP_DX10_CLAMP_OFF;
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
  return convert<rocjitsu::arith::Conversion64::DoubleToUnsigned>(flags, exec_mask,
                                                                  instruction_flags, d, a);
}

// Convert F32 to F64 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f64_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags &= ~RJ_ARITH_FP_DX10_CLAMP_OFF;
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
  return convert<rocjitsu::arith::Conversion64::FloatToDouble>(flags, exec_mask, instruction_flags,
                                                               d, a);
}

// Convert F64 to F32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f32_f64_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags &= ~RJ_ARITH_FP_DX10_CLAMP_OFF;
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
  return convert<rocjitsu::arith::Conversion64::DoubleToFloat>(flags, exec_mask, instruction_flags,
                                                               d, a);
}

// Convert I32 to F64 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f64_i32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a) {
  flags &= ~RJ_ARITH_FP_DX10_CLAMP_OFF;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return convert_wave64<rocjitsu::arith::Conversion64::SignedToDouble>(flags, exec_mask,
                                                                       instruction_flags, d, a);
}

// Convert U32 to F64 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f64_u32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a) {
  flags &= ~RJ_ARITH_FP_DX10_CLAMP_OFF;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return convert_wave64<rocjitsu::arith::Conversion64::UnsignedToDouble>(flags, exec_mask,
                                                                         instruction_flags, d, a);
}

// Convert F64 to I32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_i32_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags &= ~RJ_ARITH_FP_DX10_CLAMP_OFF;
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
  return convert_wave64<rocjitsu::arith::Conversion64::DoubleToSigned>(flags, exec_mask,
                                                                       instruction_flags, d, a);
}

// Convert F64 to U32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_u32_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags &= ~RJ_ARITH_FP_DX10_CLAMP_OFF;
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
  return convert_wave64<rocjitsu::arith::Conversion64::DoubleToUnsigned>(flags, exec_mask,
                                                                         instruction_flags, d, a);
}

// Convert F32 to F64 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f64_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags &= ~RJ_ARITH_FP_DX10_CLAMP_OFF;
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
  return convert_wave64<rocjitsu::arith::Conversion64::FloatToDouble>(flags, exec_mask,
                                                                      instruction_flags, d, a);
}

// Convert F64 to F32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f32_f64_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags &= ~RJ_ARITH_FP_DX10_CLAMP_OFF;
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
  return convert_wave64<rocjitsu::arith::Conversion64::DoubleToFloat>(flags, exec_mask,
                                                                      instruction_flags, d, a);
}

// Convert I32 to F64 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f64_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // RDNA3 ISA 7.2.3.1: output scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
  }
  return rj_arith_v_cvt_f64_i32_w32_gfx12(flags, exec_mask, instruction_flags, d, a);
}

// Convert I32 to F64 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f64_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // RDNA3 ISA 7.2.3.1: output scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
  }
  return rj_arith_v_cvt_f64_i32_w64_gfx12(flags, exec_mask, instruction_flags, d, a);
}

// Convert U32 to F64 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f64_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // RDNA3 ISA 7.2.3.1: output scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
  }
  return rj_arith_v_cvt_f64_u32_w32_gfx12(flags, exec_mask, instruction_flags, d, a);
}

// Convert U32 to F64 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f64_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // RDNA3 ISA 7.2.3.1: output scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
  }
  return rj_arith_v_cvt_f64_u32_w64_gfx12(flags, exec_mask, instruction_flags, d, a);
}

// Convert F64 to I32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_i32_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_i32_f64_w32_gfx12(flags, exec_mask, instruction_flags, d, a, nullptr);
}

// Convert F64 to I32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_i32_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_i32_f64_w64_gfx12(flags, exec_mask, instruction_flags, d, a, nullptr);
}

// Convert F64 to U32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_u32_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_u32_f64_w32_gfx12(flags, exec_mask, instruction_flags, d, a, nullptr);
}

// Convert F64 to U32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_u32_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_u32_f64_w64_gfx12(flags, exec_mask, instruction_flags, d, a, nullptr);
}

// Convert F32 to F64 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f64_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // RDNA3 ISA 7.2.3.1: output scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return convert<rocjitsu::arith::Conversion64::FloatToDouble>(
      flags, exec_mask, instruction_flags, d, a, bool(flags & RJ_ARITH_FP_IEEE_MODE));
}

// Convert F32 to F64 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f64_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // RDNA3 ISA 7.2.3.1: output scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return convert_wave64<rocjitsu::arith::Conversion64::FloatToDouble>(
      flags, exec_mask, instruction_flags, d, a, bool(flags & RJ_ARITH_FP_IEEE_MODE));
}

// Convert F64 to F32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f32_f64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // RDNA3 ISA 7.2.3.1: output scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return convert<rocjitsu::arith::Conversion64::DoubleToFloat>(flags, exec_mask, instruction_flags,
                                                               d, a);
}

// Convert F64 to F32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f32_f64_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // RDNA3 ISA 7.2.3.1: output scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_MASK);
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return convert_wave64<rocjitsu::arith::Conversion64::DoubleToFloat>(flags, exec_mask,
                                                                      instruction_flags, d, a);
}
