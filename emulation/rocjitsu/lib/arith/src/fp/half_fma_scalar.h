// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar FP16 FMA rounding, exceptional values, and FP status rules.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/bits.h"

#include "common/float_formats.h"
#include "common/internal.h"
#include "fp/fma_integer.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Whether required FP status reporting is unsupported for this FP16 FMA policy.
inline bool half_fma_reporting_unsupported(uint64_t flags, bool rdna3) {
  uint64_t semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  return semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL &&
         (rdna3 || (flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                             RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_ROUND_MASK)));
}

// Round finite FP64 bits to FP16 with guest rounding and optional finite saturation.
// Directed rounding follows rocjitsu pseudo_scalar.h::round_f64_to_f16;
// integer discarded-bit tests avoid depending on host conversions.
inline uint16_t half_fma_narrow(uint64_t bits, bool saturate, unsigned rounding = 0) {
  // Adapted from rocjitsu mixed_fma_simd.h: round_finite_f16_simd.
  int exponent = (bits >> 52) & 2047;
  uint16_t sign = uint16_t((bits >> 48) & 0x8000);
  const bool away = rounding == (sign ? 2u : 1u);
  if (exponent < 998) {
    return sign | (away && (bits & 0x7fffffffffffffffULL) ? 1 : 0);
  }
  if (exponent > 1038) {
    return sign | ((saturate || (rounding && !away)) ? 0x7bff : 0x7c00);
  }
  int shift = exponent < 1009 ? 1051 - exponent : 42;
  uint64_t significand = (bits & 0x000fffffffffffffULL) | (1ULL << 52);
  uint64_t rounded = significand >> shift;
  if (exponent >= 1009) {
    rounded += (exponent - 1009) << 10;
  }
  uint64_t remainder = significand & ((1ULL << shift) - 1);
  uint64_t halfway = 1ULL << (shift - 1);
  rounded += rounding ? (away && remainder != 0)
                      : remainder > halfway || (remainder == halfway && (rounded & 1));
  if (rounded >= 0x7c00) {
    rounded = (saturate || (rounding && !away)) ? 0x7bff : 0x7c00;
  }
  return sign | uint16_t(rounded);
}

// Classify the unrounded value using the guest-rounding-dependent FP16 tininess boundary.
inline bool half_fma_tiny(double value, uint16_t rounded, unsigned rounding) {
  bool away = rounding == ((rounded & 0x8000) ? 2u : 1u);
  double threshold = 0x1p-14 - (rounding == 0 ? 0x1p-26 : away ? 0x1p-25 : 0.0);
  uint16_t magnitude = rounded & 0x7fff;
  return magnitude < 0x400 ||
         (magnitude == 0x400 &&
          (__builtin_fabs(value) < threshold || (away && __builtin_fabs(value) == threshold)));
}

// Clamp an FP16 encoding to [0,1], mapping NaNs to zero.
inline uint16_t half_fma_clamp(uint16_t value) {
  return (value & 0x8000) || (value & 0x7fff) > 0x7c00 ? 0 : value > 0x3c00 ? 0x3c00 : value;
}

// Adapted from rocjitsu shared/fp_mode.h: fma_f16 and finish_fma_f16.
// Guest rounding, input/output denormals and architecture-specific NaN quieting.
// A null status pointer skips reporting; otherwise OR into existing bits.
// Reporting requires gfx12, nearest-even guest rounding, and preserved input/output denormals;
// callers enforce this domain with half_fma_reporting_unsupported.
template <bool Rdna3 = false>
inline uint16_t half_fma_value(uint16_t a, uint16_t b, uint16_t c, uint32_t instruction_flags,
                               uint64_t flags, uint32_t *exceptions = nullptr) {
  bool saturate = flags & RJ_ARITH_FP16_OVFL;
  uint16_t inputs[] = {a, b, c};
  for (int i = 0; i < 3; ++i) {
    if (instruction_flags & (RJ_ARITH_ALU_ABS_A << i)) {
      inputs[i] &= 0x7fff;
    }
    if (instruction_flags & (RJ_ARITH_ALU_NEG_A << i)) {
      inputs[i] ^= 0x8000;
    }
    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (inputs[i] & 0x7fff) < 0x400) {
      inputs[i] &= 0x8000;
    }
  }
  a = inputs[0];
  b = inputs[1];
  c = inputs[2];
  const bool quiet_nan = !Rdna3 || (flags & RJ_ARITH_FP_IEEE_MODE);
  uint16_t ma = a & 0x7fff, mb = b & 0x7fff, mc = c & 0x7fff;
  const bool a_nan = ma > 0x7c00, b_nan = mb > 0x7c00, c_nan = mc > 0x7c00;
  const bool nan = a_nan || b_nan || c_nan;
  const bool invalid_product = (ma == 0 && mb == 0x7c00) || (mb == 0 && ma == 0x7c00);
  const bool opposite_infinities =
      !nan && (ma == 0x7c00 || mb == 0x7c00) && mc == 0x7c00 && ((a ^ b ^ c) & 0x8000);

  // gfx1201, default MODE: CLAMP suppressed all tested exception causes,
  // unlike RDNA4 ISA 7.2.3.1's before-CLAMP rule. This is an empirical policy.
  // NaNs and invalid operations suppress
  // denormal-input reporting; infinities alone do not. Classification uses bits
  // and the integer FMA intermediate, never the host's exception flags.
  bool report = exceptions && !(instruction_flags & RJ_ARITH_ALU_CLAMP);
  if (report) {
    // Every signaling NaN contributes, regardless of which operand supplies the payload.
    bool invalid = invalid_product || (a_nan && !(ma & 0x200)) || (b_nan && !(mb & 0x200)) ||
                   (c_nan && !(mc & 0x200)) || opposite_infinities;
    if (invalid) {
      *exceptions |= RJ_ARITH_EXCEPTION_INVALID;
    } else if (!nan && ((ma && ma < 0x400) || (mb && mb < 0x400) || (mc && mc < 0x400))) {
      *exceptions |= RJ_ARITH_EXCEPTION_INPUT_DENORM;
    }
  }
  uint16_t exceptional = 0;
  if (invalid_product) {
    exceptional = 0xfe00;
  } else if (nan) {
    exceptional = (a_nan ? a : b_nan ? b : c) | (quiet_nan ? 0x0200 : 0);
  } else if (ma == 0x7c00 || mb == 0x7c00) {
    exceptional = ((a ^ b) & 0x8000) | 0x7c00;
    if (opposite_infinities) {
      exceptional = 0xfe00;
    }
  } else if (mc == 0x7c00) {
    exceptional = c;
  }
  if (exceptional) {
    const bool keep_nan =
        Rdna3 && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) && (exceptional & 0x7fff) > 0x7c00;
    return (instruction_flags & RJ_ARITH_ALU_CLAMP) && !keep_nan ? half_fma_clamp(exceptional)
                                                                 : exceptional;
  }

  uint64_t bits = fma_finite_round_odd(as_bits(f16_to_float(a)), as_bits(f16_to_float(b)),
                                       as_bits(f16_to_float(c)));
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  // The accumulator preserves matching zero signs. Only downward rounding
  // changes its +0 cancellation result to -0.
  const bool matching_zeros = (ma == 0 || mb == 0) && mc == 0 && ((a ^ b ^ c) & 0x8000) == 0;
  if (rounding == 2 && !(bits & 0x7fffffffffffffffULL) && !matching_zeros) {
    bits = 1ULL << 63;
  }
  double value = rocjitsu::arith::bit_cast<double>(bits);
  uint16_t result = half_fma_narrow(bits, false, rounding);
  uint32_t omod = (instruction_flags >> 6) & 3;
  if (report) {
    // FP16_OVFL changes the result, not overflow reporting. The round-to-odd
    // intermediate retains discarded information even across large exponent gaps.
    uint16_t rounded = rounding ? half_fma_narrow(bits, false) : result;
    uint16_t magnitude = rounded & 0x7fff;
    if (magnitude == 0x7c00) {
      *exceptions |= RJ_ARITH_EXCEPTION_OVERFLOW | (omod ? 0 : RJ_ARITH_EXCEPTION_INEXACT);
    } else if (!omod && value != double(f16_to_float(rounded))) {
      *exceptions |= RJ_ARITH_EXCEPTION_INEXACT;

      // Tininess is detected after significand rounding, before limiting the
      // exponent: the unbounded-exponent midpoint 2^-14 - 2^-26 is not tiny.
      // The FP16 encoding midpoint 2^-14 - 2^-25 is still tiny.
      if (__builtin_fabs(value) < 0x1p-14 - 0x1p-26) {
        *exceptions |= RJ_ARITH_EXCEPTION_UNDERFLOW;
      }
    }
    if ((omod == 1 || omod == 2) && magnitude < 0x7c00 && (magnitude >> 10) + omod >= 31) {
      *exceptions |= RJ_ARITH_EXCEPTION_OVERFLOW;
    }
  }
  if (saturate && (result & 0x7fff) == 0x7c00) {
    result = (result & 0x8000) | 0x7bff;
  }
  if (!omod && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    if (half_fma_tiny(value, result, rounding)) {
      result &= 0x8000;
    }
  }

  if (omod) {
    uint16_t magnitude = result & 0x7fff;
    if (half_fma_tiny(value, result, rounding)) {
      result = 0;
    } else if (omod == 3 && magnitude < 0x0800) {
      result &= 0x8000;
    } else if (magnitude < 0x7c00) {
      const double scales[] = {1, 2, 4, 0.5};
      double scaled = double(f16_to_float(result)) * scales[omod];
      result = half_fma_narrow(rocjitsu::arith::bit_cast<uint64_t>(scaled), saturate, rounding);
      if (saturate && (result & 0x7fff) == 0x7c00) {
        result = (result & 0x8000) | 0x7bff;
      }
    }
  }
  return instruction_flags & RJ_ARITH_ALU_CLAMP ? half_fma_clamp(result) : result;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
