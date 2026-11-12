// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar floating-point conversions under the default host FP environment.
// Supported rounding controls and FP status depend on the instruction.
// Owns public entry points and their validation/dispatch policy.

// Conversion models follow rocjitsu generated/shared/execute_shared.h and
// util/data_types.h, with GFX1201-verified denormal and NaN handling.

#include "common/bits.h"
#include "common/float_encoding.h"
#include "common/half_conversion.h"
#include "common/validation.h"
#include "conversion/conversion32.h"
#include "conversion/float_conversion.h"
#include "rj_arith/detail/rj_arith_scalar.h"

#include <stdint.h>

namespace {

enum class Convert {
  SignedToFloat,
  UnsignedToFloat,
  FloatToSigned,
  FloatToUnsigned,
  FloatToHalf,
  HalfToFloat,
  HighHalfToFloat,
  PackedHalfRtz
};

// Convert a scalar FP value. Reject invalid controls before changing outputs.
template <Convert Op, bool LegacyNaN = false>
int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t a, uint32_t b = 0) {
  // These conversions round once into the destination; directed rounding uses the VALU
  // integer models and nearest-even the native or software conversion.
  constexpr bool rounded =
      Op == Convert::SignedToFloat || Op == Convert::UnsignedToFloat || Op == Convert::FloatToHalf;
  if (int error = rocjitsu::arith::validate(flags, mode,
                                            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                                                (rounded ? RJ_ARITH_FP_ROUND_MASK : 0))) {
    return error;
  }
  if (!rounded &&
      (flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  if constexpr (Op == Convert::SignedToFloat || Op == Convert::UnsignedToFloat) {
    constexpr auto convert = Op == Convert::SignedToFloat
                                 ? rocjitsu::arith::Conversion32::SignedToFloat
                                 : rocjitsu::arith::Conversion32::UnsignedToFloat;
    // Native conversion rounds to nearest even, given default host FP state.
    if (rounding) {
      *d = rocjitsu::arith::conversion_detail::convert32<convert>(a, flags, 0);
    } else {
      *d = rocjitsu::arith::as_bits(Op == Convert::SignedToFloat ? float(int32_t(a)) : float(a));
    }
  } else if constexpr (Op == Convert::FloatToHalf) {
    if (rounding) {
      *d =
          rocjitsu::arith::conversion_detail::convert16<rocjitsu::arith::Conversion16::FloatToHalf>(
              a, flags, 0);
    } else {
      uint32_t x =
          (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && !(a & 0x7f800000) ? a & 0x80000000 : a;
      if (!LegacyNaN && (x & 0x7fffffff) > 0x7f800000) {
        x |= 0x400000;
      }
      uint32_t result =
          rocjitsu::arith::float_to_f16(rocjitsu::arith::as_float(x), flags & RJ_ARITH_FP16_OVFL);
      // The rounding interval just below minimum normal still counts as tiny.
      if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
          (!(result & 0x7c00) || ((result & 0x7fff) == 0x400 && (x & 0x7fffffff) < 0x387ff000))) {
        result &= 0x8000;
      }
      *d = result;
    }
  } else {
    constexpr bool half = Op == Convert::HalfToFloat || Op == Convert::HighHalfToFloat;
    if constexpr (Op == Convert::HighHalfToFloat) {
      a >>= 16;
    }
    if constexpr (half) {
      a &= 65535;
    }
    constexpr uint32_t sign = half ? 0x8000 : 0x80000000, inf = half ? 0x7c00 : 0x7f800000;
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      if (!(a & inf)) {
        a &= sign;
      }
      if (!(b & 0x7f800000)) {
        b &= 0x80000000;
      }
    }
    if (!LegacyNaN && (a & (sign - 1)) > inf) {
      a |= half ? 0x200 : 0x400000;
    }
    float x = half ? rocjitsu::arith::f16_to_float(uint16_t(a)) : rocjitsu::arith::as_float(a);
    if constexpr (Op == Convert::FloatToSigned) {
      *d = rocjitsu::arith::conversion_detail::convert32<
          rocjitsu::arith::Conversion32::FloatToSigned>(a, 0, 0);
    } else if constexpr (Op == Convert::FloatToUnsigned) {
      *d = rocjitsu::arith::conversion_detail::convert32<
          rocjitsu::arith::Conversion32::FloatToUnsigned>(a, 0, 0);
    } else if constexpr (half) {
      *d = rocjitsu::arith::as_bits(x);
    } else {
      uint32_t result = rocjitsu::arith::half_rtz<!LegacyNaN>(a) |
                        (uint32_t(rocjitsu::arith::half_rtz<!LegacyNaN>(b)) << 16);
      if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
        if (!(result & 0x7c00)) {
          result &= 0xffff8000u;
        }
        if (!(result & 0x7c000000)) {
          result &= 0x8000ffff;
        }
      }
      *d = result;
    }
  }
  return RJ_ARITH_SUCCESS;
}

// gfx1250 scalar conversions retain legacy NaN payloads and do not report FP status.
// Compatibility ignores guest denormal flushing; FP16 overflow saturation remains effective.
template <Convert Op>
int run_gfx1250(uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a, uint32_t b = 0) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate(
          flags, mode, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (compatibility) {
    flags &= ~(RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
  }
  return run<Op, true>(flags, 0, d, a, b);
}

} // namespace

// Convert I32 to F32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_f32_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
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
  return run<Convert::SignedToFloat>(flags, instruction_flags, d, a);
}

// Convert U32 to F32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_f32_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
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
  return run<Convert::UnsignedToFloat>(flags, instruction_flags, d, a);
}

// Convert F32 to I32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_i32_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
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
  return run<Convert::FloatToSigned>(flags, instruction_flags, d, a);
}

// Convert F32 to U32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_u32_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
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
  return run<Convert::FloatToUnsigned>(flags, instruction_flags, d, a);
}

// Convert F32 to F16.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_f16_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
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
  return run<Convert::FloatToHalf>(flags, instruction_flags, d, a);
}

// Convert F16 to F32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_f32_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
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
  return run<Convert::HalfToFloat>(flags, instruction_flags, d, a);
}

// Convert F16 to F32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_hi_f32_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
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
  return run<Convert::HighHalfToFloat>(flags, instruction_flags, d, a);
}

// Convert F32 to F16 and pack results with packed elements, round-to-zero conversion.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_pk_rtz_f16_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                  uint32_t a, uint32_t b,
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
  return run<Convert::PackedHalfRtz>(flags, instruction_flags, d, a, b);
}

// Convert gfx1250 scalar operands; FP status is unused and strict modes unsupported.
int rj_arith_s_cvt_f32_i32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<Convert::SignedToFloat>(flags, instruction_flags, d, a);
}

// Convert gfx1250 scalar operands; FP status is unused and strict modes unsupported.
int rj_arith_s_cvt_f32_u32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<Convert::UnsignedToFloat>(flags, instruction_flags, d, a);
}

// Convert gfx1250 scalar operands; FP status is unused and strict modes unsupported.
int rj_arith_s_cvt_i32_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<Convert::FloatToSigned>(flags, instruction_flags, d, a);
}

// Convert gfx1250 scalar operands; FP status is unused and strict modes unsupported.
int rj_arith_s_cvt_u32_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<Convert::FloatToUnsigned>(flags, instruction_flags, d, a);
}

// Convert gfx1250 scalar operands; FP status is unused and strict modes unsupported.
int rj_arith_s_cvt_f16_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<Convert::FloatToHalf>(flags, instruction_flags, d, a);
}

// Convert gfx1250 scalar operands; FP status is unused and strict modes unsupported.
int rj_arith_s_cvt_f32_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<Convert::HalfToFloat>(flags, instruction_flags, d, a);
}

// Convert gfx1250 scalar operands; FP status is unused and strict modes unsupported.
int rj_arith_s_cvt_hi_f32_f16_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                      uint32_t a, [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<Convert::HighHalfToFloat>(flags, instruction_flags, d, a);
}

// Convert gfx1250 scalar operands; FP status is unused and strict modes unsupported.
int rj_arith_s_cvt_pk_rtz_f16_f32_gfx1250(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                          uint32_t a, uint32_t b,
                                          [[maybe_unused]] uint32_t *excp_flag_user) {
  return run_gfx1250<Convert::PackedHalfRtz>(flags, instruction_flags, d, a, b);
}
