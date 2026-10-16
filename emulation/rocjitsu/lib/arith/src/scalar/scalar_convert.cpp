// SPDX-License-Identifier: MIT

// Scalar floating-point conversion with guest rounding/status rules.
// Owns public entry points and their validation/dispatch policy.

// Conversion models follow rocjitsu generated/shared/execute_shared.h and
// util/data_types.h, with GFX1201-verified denormal and NaN handling.

#include "common/bits.h"
#include "common/float_formats.h"
#include "common/half_conversion.h"
#include "common/internal.h"
#include "conversion/conversion32.h"
#include "rj_arith/rj_arith.h"

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
template <Convert Op>
int run(uint64_t flags, uint32_t mode, uint32_t *d, uint32_t a, uint32_t b = 0) {
  if (int error = rocjitsu::arith::validate(flags, mode, false,
                                            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  constexpr bool from_integer = Op == Convert::SignedToFloat || Op == Convert::UnsignedToFloat;
  if constexpr (from_integer) {
    if constexpr (Op == Convert::SignedToFloat) {
      *d = rocjitsu::arith::as_bits(float(rocjitsu::arith::extend_integer<32, true>(a)));
    } else {
      *d = rocjitsu::arith::as_bits(float(a));
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
    if ((a & (sign - 1)) > inf) {
      a |= half ? 0x200 : 0x400000;
    }
    float x = half ? rocjitsu::arith::f16_to_float(uint16_t(a)) : rocjitsu::arith::as_float(a);
    if constexpr (Op == Convert::FloatToSigned) {
      *d = rocjitsu::arith::truncate_integer<true>(x);
    } else if constexpr (Op == Convert::FloatToUnsigned) {
      *d = rocjitsu::arith::truncate_integer<false>(x);
    } else if constexpr (half) {
      *d = rocjitsu::arith::as_bits(x);
    } else {
      uint32_t result;
      if constexpr (Op == Convert::FloatToHalf) {
        result = rocjitsu::arith::float_to_f16(x, flags & RJ_ARITH_FP16_OVFL);
      } else {
        result = rocjitsu::arith::half_rtz(a) | (uint32_t(rocjitsu::arith::half_rtz(b)) << 16);
      }
      if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
        bool tiny = !(result & 0x7c00);
        if constexpr (Op == Convert::FloatToHalf) {
          // The rounding interval just below minimum normal still counts as tiny for this
          // conversion.
          tiny |= (result & 0x7fff) == 0x400 && (a & 0x7fffffff) < 0x387ff000;
        }
        if (tiny) {
          result &= 0xffff8000u;
        }
        if constexpr (Op == Convert::PackedHalfRtz) {
          if (!(result & 0x7c000000)) {
            result &= 0x8000ffff;
          }
        }
      }
      *d = result;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Convert I32 to F32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_f32_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
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
  return run<Convert::SignedToFloat>(flags, instruction_flags, d, a);
}

// Convert U32 to F32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_f32_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
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
