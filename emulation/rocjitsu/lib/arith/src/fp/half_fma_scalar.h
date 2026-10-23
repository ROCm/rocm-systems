// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar gfx12 FP16 FMA with default guest controls, optional finite saturation,
// and FP-status reporting.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "rj_arith/rj_arith.h"

#include "common/float_formats.h"
#include "common/internal.h"
#include "fp/fma_integer.h"

#include <bit>
#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith {

// Round finite FP64 bits to nearest-even FP16.
inline uint16_t half_fma_narrow(uint64_t bits) {
  // Adapted from rocjitsu mixed_fma_simd.h: round_finite_f16_simd.
  int exponent = (bits >> 52) & 2047;
  uint16_t sign = uint16_t((bits >> 48) & 0x8000);
  if (exponent < 998) {
    return sign;
  }
  if (exponent > 1038) {
    return sign | 0x7c00;
  }
  int shift = exponent < 1009 ? 1051 - exponent : 42;
  uint64_t significand = (bits & 0x000fffffffffffffULL) | (1ULL << 52);
  uint64_t rounded = significand >> shift;
  if (exponent >= 1009) {
    rounded += (exponent - 1009) << 10;
  }
  uint64_t remainder = significand & ((1ULL << shift) - 1);
  uint64_t halfway = 1ULL << (shift - 1);
  rounded += remainder > halfway || (remainder == halfway && (rounded & 1));
  if (rounded >= 0x7c00) {
    rounded = 0x7c00;
  }
  return sign | uint16_t(rounded);
}

// Adapted from rocjitsu shared/fp_mode.h: fma_f16 and finish_fma_f16.
// Default guest rounding and denormal policy; preserve architecture-specific NaN results.
// A null status pointer skips reporting; otherwise OR causes into existing bits.
// Reporting is supported for gfx12 only.
inline uint16_t half_fma_value(uint16_t a, uint16_t b, uint16_t c, bool saturate,
                               uint32_t *exceptions = nullptr) {
  uint16_t ma = a & 0x7fff, mb = b & 0x7fff, mc = c & 0x7fff;
  const bool a_nan = ma > 0x7c00, b_nan = mb > 0x7c00, c_nan = mc > 0x7c00;
  const bool nan = a_nan || b_nan || c_nan;
  const bool invalid_product = (ma == 0 && mb == 0x7c00) || (mb == 0 && ma == 0x7c00);
  const bool opposite_infinities =
      !nan && (ma == 0x7c00 || mb == 0x7c00) && mc == 0x7c00 && ((a ^ b ^ c) & 0x8000);

  // NaNs and invalid operations suppress
  // denormal-input reporting; infinities alone do not. Classification uses bits
  // and the integer FMA intermediate, never the host's exception flags.
  bool report = exceptions != nullptr;
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
    exceptional = (a_nan ? a : b_nan ? b : c) | 0x0200;
  } else if (ma == 0x7c00 || mb == 0x7c00) {
    exceptional = ((a ^ b) & 0x8000) | 0x7c00;
    if (opposite_infinities) {
      exceptional = 0xfe00;
    }
  } else if (mc == 0x7c00) {
    exceptional = c;
  }
  if (exceptional) {
    return exceptional;
  }

  uint64_t bits = fma_finite_round_odd(as_bits(f16_to_float(a)), as_bits(f16_to_float(b)),
                                       as_bits(f16_to_float(c)));
  uint16_t result = half_fma_narrow(bits);
  if (report) {
    double value = std::bit_cast<double>(bits);
    // FP16_OVFL changes the result, not overflow reporting. The round-to-odd
    // intermediate retains discarded information even across large exponent gaps.
    uint16_t magnitude = result & 0x7fff;
    if (magnitude == 0x7c00) {
      *exceptions |= RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT;
    } else if (value != double(f16_to_float(result))) {
      *exceptions |= RJ_ARITH_EXCEPTION_INEXACT;

      // Tininess is detected after significand rounding, before limiting the
      // exponent: the unbounded-exponent midpoint 2^-14 - 2^-26 is not tiny.
      // The FP16 encoding midpoint 2^-14 - 2^-25 is still tiny.
      if (std::abs(value) < 0x1p-14 - 0x1p-26) {
        *exceptions |= RJ_ARITH_EXCEPTION_UNDERFLOW;
      }
    }
  }
  return saturate && (result & 0x7fff) == 0x7c00 ? (result & 0x8000) | 0x7bff : result;
}

} // namespace rocjitsu::arith
