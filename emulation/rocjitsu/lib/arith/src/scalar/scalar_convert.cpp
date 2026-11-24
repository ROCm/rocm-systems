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
#include "fp/rounding_status.h"
#include "rj_arith/detail/rj_arith_scalar.h"
#include "scalar/fp_status.h"
#include "scalar/gfx1150_policy.h"

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

// Reproduce the scalar conversion causes, without host FP-state changes. gfx12 uses the
// shared hardware rule; Inherited keeps gfx1150 on the stored-result rule.
// Packed RTZ conversion deliberately leaves FP status unchanged on gfx12.
template <Convert Op, bool FractionalInexact = false, bool Inherited = false>
uint32_t conversion_causes(uint64_t flags, uint32_t a, uint32_t result) {
  using namespace rocjitsu::arith;
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  if constexpr (Op == Convert::PackedHalfRtz) {
    return 0;
  } else if constexpr (Op == Convert::SignedToFloat || Op == Convert::UnsignedToFloat) {
    // Every 32-bit integer is exact in FP64; only INEXACT can occur.
    const double exact = Op == Convert::SignedToFloat ? double(int32_t(a)) : double(a);
    if constexpr (Inherited) {
      return scalar_result_causes<false>(exact, result, rounding);
    } else {
      return a ? rounding_causes<false>(double_bits(exact), result, rounding) : 0;
    }
  } else {
    constexpr bool half = Op == Convert::HalfToFloat || Op == Convert::HighHalfToFloat;
    constexpr uint32_t normal = half ? 0x400 : 0x800000;
    constexpr uint32_t sign = half ? 0x8000 : 0x80000000;
    if constexpr (Op == Convert::HighHalfToFloat) {
      a >>= 16;
    }
    a &= sign | (sign - 1);
    uint32_t causes = scalar_input_causes<half>(a, flags);
    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (a & (sign - 1)) < normal) {
      a &= sign;
    }
    const double exact = scalar_fp_value<half>(a);
    if constexpr (Op == Convert::FloatToSigned || Op == Convert::FloatToUnsigned) {
      constexpr double low = Op == Convert::FloatToSigned ? -0x1p31 : 0;
      constexpr double high = Op == Convert::FloatToSigned ? 0x1p31 - 1 : 0x1p32 - 1;
      if (!__builtin_isfinite(exact) || exact < low || __builtin_trunc(exact) > high) {
        return RJ_ARITH_EXCEPTION_INVALID;
      }
      // gfx1150 reports discarded fractional bits; gfx12 suppresses this cause.
      if constexpr (FractionalInexact) {
        if (__builtin_trunc(exact) != exact) {
          causes |= RJ_ARITH_EXCEPTION_INEXACT;
        }
      }
    } else if constexpr (Inherited) {
      if (__builtin_isfinite(exact)) {
        causes |= scalar_result_causes<!half>(exact, result, rounding);
      }
    } else if (__builtin_isfinite(exact) && exact != 0) {
      causes |= rounding_causes<!half>(double_bits(exact), result, rounding);
    }
    return causes;
  }
}

// Compatibility and explicit mode 3 share the gfx1201 hardware-fitted scalar conversion status.
template <Convert Op>
int run_policy(uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a, uint32_t b,
               uint32_t *status) {
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  const bool report = rocjitsu::arith::reports_fp_status(flags);
  if (mode >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  const uint64_t numeric_flags = rocjitsu::arith::without_fp_status(flags);
  int error = run<Op>(numeric_flags, uint32_t(mode), d, a, b);
  if constexpr (Op != Convert::PackedHalfRtz) {
    if (!error && (compatibility || report)) {
      *status |= conversion_causes<Op>(numeric_flags, a, *d);
    }
  }
  return error;
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

// Retain gfx1150 conversion rounding, IEEE NaNs and both packed-input causes.
template <Convert Op>
int run_gfx1150(uint64_t flags, uint64_t mode, uint32_t *d, uint32_t a, uint32_t b,
                uint32_t *status) {
  using namespace rocjitsu::arith;
  using namespace conversion_detail;
  const bool compatibility = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  if (int error = prepare_gfx1150_scalar(flags)) {
    return error;
  }
  constexpr bool rounded =
      Op == Convert::SignedToFloat || Op == Convert::UnsignedToFloat || Op == Convert::FloatToHalf;
  if (int error = validate(flags, mode,
                           RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                               RJ_ARITH_FP_IEEE_MODE | (rounded ? RJ_ARITH_FP_ROUND_MASK : 0))) {
    return error;
  }
  const bool quiet = flags & RJ_ARITH_FP_IEEE_MODE;
  if constexpr (Op == Convert::SignedToFloat || Op == Convert::UnsignedToFloat) {
    using From = std::conditional_t<Op == Convert::SignedToFloat, model::I32, model::U32>;
    *d = float_policy<From, fmt::F32>(flags, 0, quiet)(a);
  } else if constexpr (Op == Convert::FloatToHalf) {
    *d = float_policy<fmt::F32, fmt::F16>(flags, 0, quiet)(a);
  } else if constexpr (Op == Convert::HalfToFloat || Op == Convert::HighHalfToFloat) {
    const uint32_t input = Op == Convert::HighHalfToFloat ? a >> 16 : a;
    *d = float_policy<fmt::F16, fmt::F32>(flags, 0, quiet)(input & 0xffff);
  } else if constexpr (Op == Convert::PackedHalfRtz) {
    flags = (flags & ~RJ_ARITH_FP_ROUND_MASK) | RJ_ARITH_FP_ROUND_ZERO;
    auto policy = float_policy<fmt::F32, fmt::F16>(flags, 0, quiet);
    *d = policy(a) | (policy(b) << 16);
  } else {
    // Integer destinations use the same truncating, saturating raw-bit policy.
    if (int error = run<Op>(flags & ~RJ_ARITH_FP_IEEE_MODE, uint32_t(mode), d, a, b)) {
      return error;
    }
  }
  if (compatibility) {
    if constexpr (Op == Convert::PackedHalfRtz) {
      *status |= conversion_causes<Convert::FloatToHalf, false, true>(flags, a, *d & 0xffff) |
                 conversion_causes<Convert::FloatToHalf, false, true>(flags, b, *d >> 16);
    } else {
      *status |= conversion_causes<Op, true, true>(flags, a, *d);
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Convert I32 to F32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_f32_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t *excp_flag_user) {
  return run_policy<Convert::SignedToFloat>(flags, instruction_flags, d, a, 0, excp_flag_user);
}

// Convert U32 to F32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_f32_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t *excp_flag_user) {
  return run_policy<Convert::UnsignedToFloat>(flags, instruction_flags, d, a, 0, excp_flag_user);
}

// Convert F32 to I32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_i32_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t *excp_flag_user) {
  return run_policy<Convert::FloatToSigned>(flags, instruction_flags, d, a, 0, excp_flag_user);
}

// Convert F32 to U32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_u32_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t *excp_flag_user) {
  return run_policy<Convert::FloatToUnsigned>(flags, instruction_flags, d, a, 0, excp_flag_user);
}

// Convert F32 to F16.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_f16_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t *excp_flag_user) {
  return run_policy<Convert::FloatToHalf>(flags, instruction_flags, d, a, 0, excp_flag_user);
}

// Convert F16 to F32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_f32_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t *excp_flag_user) {
  return run_policy<Convert::HalfToFloat>(flags, instruction_flags, d, a, 0, excp_flag_user);
}

// Convert F16 to F32.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_hi_f32_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                              uint32_t *excp_flag_user) {
  return run_policy<Convert::HighHalfToFloat>(flags, instruction_flags, d, a, 0, excp_flag_user);
}

// Convert F32 to F16 and pack results with packed elements, round-to-zero conversion.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_s_cvt_pk_rtz_f16_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                  uint32_t a, uint32_t b, uint32_t *excp_flag_user) {
  return run_policy<Convert::PackedHalfRtz>(flags, instruction_flags, d, a, b, excp_flag_user);
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

// Convert gfx1150 scalar operands with explicit guest controls and inherited status.
int rj_arith_s_cvt_f32_i32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t *fp_status) {
  return run_gfx1150<Convert::SignedToFloat>(flags, instruction_flags, d, a, 0, fp_status);
}

// Convert gfx1150 scalar operands with explicit guest controls and inherited status.
int rj_arith_s_cvt_f32_u32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t *fp_status) {
  return run_gfx1150<Convert::UnsignedToFloat>(flags, instruction_flags, d, a, 0, fp_status);
}

// Convert gfx1150 scalar operands with explicit guest controls and inherited status.
int rj_arith_s_cvt_i32_f32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t *fp_status) {
  return run_gfx1150<Convert::FloatToSigned>(flags, instruction_flags, d, a, 0, fp_status);
}

// Convert gfx1150 scalar operands with explicit guest controls and inherited status.
int rj_arith_s_cvt_u32_f32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t *fp_status) {
  return run_gfx1150<Convert::FloatToUnsigned>(flags, instruction_flags, d, a, 0, fp_status);
}

// Convert gfx1150 scalar operands with explicit guest controls and inherited status.
int rj_arith_s_cvt_f16_f32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t *fp_status) {
  return run_gfx1150<Convert::FloatToHalf>(flags, instruction_flags, d, a, 0, fp_status);
}

// Convert gfx1150 scalar operands with explicit guest controls and inherited status.
int rj_arith_s_cvt_f32_f16_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t *fp_status) {
  return run_gfx1150<Convert::HalfToFloat>(flags, instruction_flags, d, a, 0, fp_status);
}

// Convert gfx1150 scalar operands with explicit guest controls and inherited status.
int rj_arith_s_cvt_hi_f32_f16_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                      uint32_t a, uint32_t *fp_status) {
  return run_gfx1150<Convert::HighHalfToFloat>(flags, instruction_flags, d, a, 0, fp_status);
}

// Convert gfx1150 scalar operands with explicit guest controls and inherited status.
int rj_arith_s_cvt_pk_rtz_f16_f32_gfx1150(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                          uint32_t a, uint32_t b, uint32_t *fp_status) {
  return run_gfx1150<Convert::PackedHalfRtz>(flags, instruction_flags, d, a, b, fp_status);
}
