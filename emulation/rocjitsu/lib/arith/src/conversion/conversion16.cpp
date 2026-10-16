// SPDX-License-Identifier: MIT

// Conversions between FP16 and FP32 or 16-bit integers.
// Owns public entry points and their validation/dispatch policy.

#include "conversion/conversion16.h"
#include "common/alu.h"
#include "common/float_formats.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace {

// Narrow to FP16 and apply output scaling/CLAMP, with optional finite-overflow saturation.
uint16_t half_output(float value, bool saturate, uint32_t mode) {
  // Round to half before scaling, with the captured RDNA4 tininess rule.
  uint16_t result = rocjitsu::arith::float_to_f16(value, saturate);
  unsigned omod = (mode >> 6) & 3;
  if (omod) {
    // 0x0400 is the smallest normal half. At that boundary the tininess
    // threshold is 2^-14 - 2^-26, encoded as FP32 0x387ff000.
    uint16_t magnitude = result & 0x7fff;
    if (magnitude < 0x400 ||
        (magnitude == 0x400 && (rocjitsu::arith::as_bits(value) & 0x7fffffff) < 0x387ff000)) {
      result = 0;
    } else if (omod == 3 && magnitude < 0x800) {
      result &= 0x8000;
    } else {
      const float scales[] = {1, 2, 4, 0.5f};
      result = rocjitsu::arith::float_to_f16(rocjitsu::arith::f16_to_float(result) * scales[omod],
                                             saturate);
    }
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    if ((result & 0x8000) || (result & 0x7fff) > 0x7c00) {
      result = 0;
    } else if (result > 0x3c00) {
      result = 0x3c00;
    }
  }
  return result;
}

// Convert FP16, FP32, or 16-bit integers for the selected lanes. Reject invalid controls
// before changing outputs.
template <rocjitsu::arith::Conversion16 Op, bool GuestWidening = false>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, bool quiet_nan = true) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return convert<Op, GuestWidening>(flags, exec_mask, uint32_t(mode), d, source, quiet_nan);
        });
  }
  constexpr bool from_integer = Op == rocjitsu::arith::Conversion16::SignedToHalf ||
                                Op == rocjitsu::arith::Conversion16::UnsignedToHalf;
  constexpr bool to_half = from_integer || Op == rocjitsu::arith::Conversion16::FloatToHalf;
  const uint32_t known =
      RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP |
      (from_integer ? 0 : RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A) |
      (Op == rocjitsu::arith::Conversion16::FloatToHalf ? 0 : RJ_ARITH_ALU_HIGH_A) |
      (Op == rocjitsu::arith::Conversion16::HalfToFloat ? 0 : RJ_ARITH_ALU_HIGH_D);
  if (int error =
          rocjitsu::arith::validate(flags, mode & ~known, false,
                                    RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                        RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                                        (GuestWidening ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  if constexpr (Op == rocjitsu::arith::Conversion16::HalfToFloat && !GuestWidening) {
    if (!quiet_nan || (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)) {
      return convert<Op, true>(flags, exec_mask, mode, d, a, quiet_nan);
    }
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    if constexpr (GuestWidening) {
      rocjitsu::arith::conversion16_widen_x86_64_v4(flags | (quiet_nan ? RJ_ARITH_FP_IEEE_MODE : 0),
                                                    exec_mask, mode, d[0], a[0]);
    } else {
      rocjitsu::arith::conversion16_x86_64_v4<Op>(flags & RJ_ARITH_FP16_OVFL, exec_mask, mode, d[0],
                                                  a[0]);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    if constexpr (GuestWidening) {
      rocjitsu::arith::conversion16_widen_x86_64_v3(flags | (quiet_nan ? RJ_ARITH_FP_IEEE_MODE : 0),
                                                    exec_mask, mode, d[0], a[0]);
    } else {
      rocjitsu::arith::conversion16_x86_64_v3<Op>(flags & RJ_ARITH_FP16_OVFL, exec_mask, mode, d[0],
                                                  a[0]);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
  unsigned sa = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0, sd = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t raw = a[0][lane];
    float value;
    if constexpr (from_integer) {
      uint16_t selected = uint16_t(raw >> sa);
      value = Op == rocjitsu::arith::Conversion16::SignedToHalf
                  ? float(int(selected) - ((selected >> 15) ? 65536 : 0))
                  : float(selected);
    } else {
      if constexpr (Op != rocjitsu::arith::Conversion16::FloatToHalf) {
        raw = rocjitsu::arith::as_bits(rocjitsu::arith::f16_to_float(uint16_t(raw >> sa)));
      }
      if constexpr (GuestWidening) {
        // rocjitsu fp_mode::cvt_f32_f16 applies the half input-denormal policy.
        if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (raw & 0x7fffffff) < 0x38800000) {
          raw &= 0x80000000;
        }
      }
      if ((!GuestWidening || quiet_nan) && (raw & 0x7fffffff) > 0x7f800000) {
        raw |= 0x400000;
      }
      value = rocjitsu::arith::alu_input(raw, mode);
    }
    if constexpr (to_half) {
      result[lane] = half_output(value, flags & RJ_ARITH_FP16_OVFL, mode);
    } else if constexpr (Op == rocjitsu::arith::Conversion16::HalfToFloat) {
      // rocjitsu apply_omod_f32 preserves NaN encodings; DX10_CLAMP
      // alone determines whether a NaN is replaced by zero when clamping.
      const uint32_t input_bits = rocjitsu::arith::as_bits(value);
      if constexpr (GuestWidening) {
        if ((input_bits & 0x7fffffff) > 0x7f800000) {
          result[lane] =
              (mode & RJ_ARITH_ALU_CLAMP) && !(flags & RJ_ARITH_FP_DX10_CLAMP_OFF) ? 0 : input_bits;
          continue;
        }
      }
      value = rocjitsu::arith::alu_output(value, mode);
      if ((mode & RJ_ARITH_ALU_OMOD_HALF) && value == 0) {
        value = 0;
      }
      result[lane] = rocjitsu::arith::as_bits(value);
    } else if constexpr (Op == rocjitsu::arith::Conversion16::HalfToUnsigned) {
      result[lane] = !(value > 0) ? 0 : value >= 65536 ? 65535 : uint16_t(value);
    } else {
      result[lane] = std::isnan(value) ? 0
                     : value >= 32768  ? 32767
                     : value <= -32768 ? 32768
                                       : uint16_t(int16_t(value));
    }
    if constexpr (Op != rocjitsu::arith::Conversion16::HalfToFloat) {
      result[lane] = (d[0][lane] & ~(uint32_t(65535) << sd)) | (result[lane] << sd);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Convert FP16, FP32, or 16-bit integers for Wave64. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::Conversion16 Op, bool GuestWidening = false>
int convert_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, bool quiet_nan = true) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = convert<Op, GuestWidening>(flags, 0, mode, nullptr, nullptr, quiet_nan)) {
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
    if (int error = convert<Op, GuestWidening>(flags, uint32_t(exec_mask >> (half * 32)), half_mode,
                                               &dest, &source, quiet_nan)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Convert I16 to F16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f16_i16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert<rocjitsu::arith::Conversion16::SignedToHalf>(flags, exec_mask, instruction_flags,
                                                              d, a);
}

// Convert U16 to F16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f16_u16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert<rocjitsu::arith::Conversion16::UnsignedToHalf>(flags, exec_mask, instruction_flags,
                                                                d, a);
}

// Convert F16 to I16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_i16_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert<rocjitsu::arith::Conversion16::HalfToSigned>(flags, exec_mask, instruction_flags,
                                                              d, a);
}

// Convert F16 to U16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_u16_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert<rocjitsu::arith::Conversion16::HalfToUnsigned>(flags, exec_mask, instruction_flags,
                                                                d, a);
}

// Convert F32 to F16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f16_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert<rocjitsu::arith::Conversion16::FloatToHalf>(flags, exec_mask, instruction_flags, d,
                                                             a);
}

// Convert F16 to F32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f32_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert<rocjitsu::arith::Conversion16::HalfToFloat>(flags, exec_mask, instruction_flags, d,
                                                             a);
}

// Convert I16 to F16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f16_i16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert_wave64<rocjitsu::arith::Conversion16::SignedToHalf>(flags, exec_mask,
                                                                     instruction_flags, d, a);
}

// Convert U16 to F16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f16_u16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert_wave64<rocjitsu::arith::Conversion16::UnsignedToHalf>(flags, exec_mask,
                                                                       instruction_flags, d, a);
}

// Convert F16 to I16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_i16_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert_wave64<rocjitsu::arith::Conversion16::HalfToSigned>(flags, exec_mask,
                                                                     instruction_flags, d, a);
}

// Convert F16 to U16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_u16_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert_wave64<rocjitsu::arith::Conversion16::HalfToUnsigned>(flags, exec_mask,
                                                                       instruction_flags, d, a);
}

// Convert F32 to F16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f16_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert_wave64<rocjitsu::arith::Conversion16::FloatToHalf>(flags, exec_mask,
                                                                    instruction_flags, d, a);
}

// Convert F16 to F32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f32_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert_wave64<rocjitsu::arith::Conversion16::HalfToFloat>(flags, exec_mask,
                                                                    instruction_flags, d, a);
}

// Convert I16 to F16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f16_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return convert<rocjitsu::arith::Conversion16::SignedToHalf>(flags, exec_mask, instruction_flags,
                                                              d, a);
}

// Convert I16 to F16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f16_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return convert_wave64<rocjitsu::arith::Conversion16::SignedToHalf>(flags, exec_mask,
                                                                     instruction_flags, d, a);
}

// Convert U16 to F16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f16_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return convert<rocjitsu::arith::Conversion16::UnsignedToHalf>(flags, exec_mask, instruction_flags,
                                                                d, a);
}

// Convert U16 to F16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f16_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return convert_wave64<rocjitsu::arith::Conversion16::UnsignedToHalf>(flags, exec_mask,
                                                                       instruction_flags, d, a);
}

// Convert F16 to I16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_i16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert<rocjitsu::arith::Conversion16::HalfToSigned>(flags, exec_mask, instruction_flags,
                                                              d, a);
}

// Convert F16 to I16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_i16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert_wave64<rocjitsu::arith::Conversion16::HalfToSigned>(flags, exec_mask,
                                                                     instruction_flags, d, a);
}

// Convert F16 to U16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_u16_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert<rocjitsu::arith::Conversion16::HalfToUnsigned>(flags, exec_mask, instruction_flags,
                                                                d, a);
}

// Convert F16 to U16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_u16_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return convert_wave64<rocjitsu::arith::Conversion16::HalfToUnsigned>(flags, exec_mask,
                                                                       instruction_flags, d, a);
}

// Convert F32 to F16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return convert<rocjitsu::arith::Conversion16::FloatToHalf>(flags, exec_mask, instruction_flags, d,
                                                             a);
}

// Convert F32 to F16 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return convert_wave64<rocjitsu::arith::Conversion16::FloatToHalf>(flags, exec_mask,
                                                                    instruction_flags, d, a);
}

// Convert F16 to F32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f32_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return convert<rocjitsu::arith::Conversion16::HalfToFloat, true>(
      flags, exec_mask, instruction_flags, d, a, (flags & RJ_ARITH_FP_IEEE_MODE) != 0);
}

// Convert F16 to F32 for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_cvt_f32_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }

  // rocjitsu fp_mode::effective_omod: RDNA3 scaling requires output flush and IEEE=0.
  if ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
  }
  return convert_wave64<rocjitsu::arith::Conversion16::HalfToFloat, true>(
      flags, exec_mask, instruction_flags, d, a, (flags & RJ_ARITH_FP_IEEE_MODE) != 0);
}
