// SPDX-License-Identifier: MIT

// Conversions between FP32 and signed/unsigned 32-bit integers.
// Owns public entry points and their validation/dispatch policy.

#include "conversion/conversion32.h"
#include "common/alu.h"
#include "common/bits.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace {

// Convert FP32 and 32-bit integers for the selected lanes. Reject invalid controls before
// changing outputs.
template <rocjitsu::arith::Conversion32 Op>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return convert<Op>(flags, exec_mask, uint32_t(mode), d, source);
        });
  }
  constexpr bool to_float = Op == rocjitsu::arith::Conversion32::SignedToFloat ||
                            Op == rocjitsu::arith::Conversion32::UnsignedToFloat;
  const uint32_t known =
      RJ_ARITH_ALU_CLAMP | (to_float ? 0 : RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A) |
      (Op == rocjitsu::arith::Conversion32::Nearest || Op == rocjitsu::arith::Conversion32::Floor
           ? 0
           : RJ_ARITH_ALU_OMOD_HALF);
  if (int error =
          rocjitsu::arith::validate(flags, mode & ~known, false,
                                    RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                        RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::conversion32_x86_64_v4<Op>(exec_mask, mode, d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::conversion32_x86_64_v3<Op>(exec_mask, mode, d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t raw = a[0][lane];
    if constexpr (to_float) {
      float x;
      if constexpr (Op == rocjitsu::arith::Conversion32::SignedToFloat) {
        x = float(rocjitsu::arith::extend_integer<32, true>(raw));
      } else {
        x = float(raw);
      }
      result[lane] = rocjitsu::arith::as_bits(rocjitsu::arith::alu_output(x, mode));
    } else {
      // Saturation and nearest ties toward +infinity follow rocjitsu's
      // codegen/execute/sema_lower.py conversion models. Classify before
      // casting: out-of-range floating-to-integer casts are undefined in C++.
      float x = rocjitsu::arith::alu_input(raw, mode);
      uint32_t nan_result = 0;
      if constexpr (Op == rocjitsu::arith::Conversion32::Nearest ||
                    Op == rocjitsu::arith::Conversion32::Floor) {
        nan_result =
            (rocjitsu::arith::as_bits(x) >> 31) ? uint32_t(INT32_MIN) : uint32_t(INT32_MAX);
      }
      if constexpr (Op == rocjitsu::arith::Conversion32::Nearest) {
        float rounded = std::floor(x);
        if (x - rounded >= 0.5f) {
          rounded += 1;
        }
        x = rounded;
      }
      if constexpr (Op == rocjitsu::arith::Conversion32::Floor) {
        x = std::floor(x);
      }
      result[lane] =
          rocjitsu::arith::truncate_integer<Op != rocjitsu::arith::Conversion32::FloatToUnsigned>(
              x, nan_result);

      // Integer CLAMP and the truncating opcodes' OMOD do not scale or
      // clamp numeric results. GPU exception reporting is not modeled.
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Convert FP32 and 32-bit integers for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::Conversion32 Op>
int convert_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = convert<Op>(flags, 0, mode, nullptr, nullptr)) {
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
    if (int error =
            convert<Op>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, &source)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Convert I32 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_i32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<rocjitsu::arith::Conversion32::SignedToFloat>(flags, exec_mask, instruction_flags,
                                                               d, a);
}

// Convert U32 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_u32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<rocjitsu::arith::Conversion32::UnsignedToFloat>(flags, exec_mask,
                                                                 instruction_flags, d, a);
}

// Convert F32 to I32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_i32_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert<rocjitsu::arith::Conversion32::FloatToSigned>(flags, exec_mask, instruction_flags,
                                                               d, a);
}

// Convert F32 to U32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_u32_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert<rocjitsu::arith::Conversion32::FloatToUnsigned>(flags, exec_mask,
                                                                 instruction_flags, d, a);
}

// Convert F32 to I32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_nearest_i32_f32_w32(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<rocjitsu::arith::Conversion32::Nearest>(flags, exec_mask, instruction_flags, d, a);
}

// Convert F32 to I32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_floor_i32_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert<rocjitsu::arith::Conversion32::Floor>(flags, exec_mask, instruction_flags, d, a);
}

// Convert I32 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_i32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<rocjitsu::arith::Conversion32::SignedToFloat>(flags, exec_mask,
                                                                      instruction_flags, d, a);
}

// Convert U32 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_u32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<rocjitsu::arith::Conversion32::UnsignedToFloat>(flags, exec_mask,
                                                                        instruction_flags, d, a);
}

// Convert F32 to I32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_i32_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert_wave64<rocjitsu::arith::Conversion32::FloatToSigned>(flags, exec_mask,
                                                                      instruction_flags, d, a);
}

// Convert F32 to U32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_u32_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert_wave64<rocjitsu::arith::Conversion32::FloatToUnsigned>(flags, exec_mask,
                                                                        instruction_flags, d, a);
}

// Convert F32 to I32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_nearest_i32_f32_w64(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<rocjitsu::arith::Conversion32::Nearest>(flags, exec_mask, instruction_flags,
                                                                d, a);
}

// Convert F32 to I32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_floor_i32_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return convert_wave64<rocjitsu::arith::Conversion32::Floor>(flags, exec_mask, instruction_flags,
                                                              d, a);
}

// Convert I32 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod for an ordinary RDNA3 FP32 result.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_f32_i32_w32_gfx12(flags, exec_mask, instruction_flags, d, a);
}

// Convert I32 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod for an ordinary RDNA3 FP32 result.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_f32_i32_w64_gfx12(flags, exec_mask, instruction_flags, d, a);
}

// Convert U32 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod for an ordinary RDNA3 FP32 result.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_f32_u32_w32_gfx12(flags, exec_mask, instruction_flags, d, a);
}

// Convert U32 to F32 for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cvt_f32_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }

  // rocjitsu fp_mode::effective_omod for an ordinary RDNA3 FP32 result.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return rj_arith_v_cvt_f32_u32_w64_gfx12(flags, exec_mask, instruction_flags, d, a);
}

// Convert F32 to I32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_i32_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_i32_f32_w32_gfx12(flags, exec_mask, instruction_flags, d, a, nullptr);
}

// Convert F32 to I32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_i32_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_i32_f32_w64_gfx12(flags, exec_mask, instruction_flags, d, a, nullptr);
}

// Convert F32 to U32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_u32_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_u32_f32_w32_gfx12(flags, exec_mask, instruction_flags, d, a, nullptr);
}

// Convert F32 to U32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_u32_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return rj_arith_v_cvt_u32_f32_w64_gfx12(flags, exec_mask, instruction_flags, d, a, nullptr);
}
