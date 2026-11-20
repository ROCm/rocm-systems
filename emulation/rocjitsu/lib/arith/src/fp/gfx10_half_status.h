// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited gfx10 FP16 Compatibility classifiers. These inspect prepared inputs
// before destination writes and leave host FP controls unchanged.

#pragma once

#include "common/bits.h"
#include "common/float_encoding.h"
#include "conversion/model/conversion.h"
#include "fp/binary.h"
#include "fp/half_fma_scalar.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Classify scaling of the rounded half before saturation or CLAMP masks its causes.
inline uint32_t gfx10_half_output_causes(double exact, uint16_t arithmetic, bool saturate,
                                         unsigned omod) {
  uint16_t magnitude = arithmetic & 0x7fff;
  if (magnitude == 0 && exact != 0) {
    return RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT;
  }
  if (magnitude == 0x7c00 && saturate) {
    magnitude = 0x7bff;
  }
  const unsigned exponent = magnitude >> 10;
  if (exponent != 0 && exponent != 31) {
    if ((omod == 1 || omod == 2) && exponent + omod >= 31) {
      return RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT;
    }
    if (omod == 3 && exponent == 1) {
      return RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT;
    }
  }
  return 0;
}

// Return the legacy per-wave causes for valid FP16 controls and selected lanes.
// gfx1010 used the ADD-only classifier; gfx1030 classified all four operations.
template <Binary Op, bool Gfx1010>
uint32_t gfx10_half_causes(uint64_t flags, uint32_t exec_mask, uint32_t modifiers,
                           const uint32_t *a, const uint32_t *b) {
  namespace cm = conversion_model;
  using F = cm::fp_format::F16;
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  const cm::conversion::FloatRounding policy{cm::conversion::mode_rounding(rounding)};
  const unsigned omod = (modifiers & RJ_ARITH_ALU_OMOD_MASK) / RJ_ARITH_ALU_OMOD_2;
  auto source = [&](uint32_t raw, uint64_t absolute, uint64_t negate) {
    uint16_t x = uint16_t(raw);
    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (x & F::kMagnitude) < F::kMinNormal) {
      x &= F::kSign;
    }
    if (modifiers & absolute) {
      x &= F::kMagnitude;
    }
    if (modifiers & negate) {
      x ^= F::kSign;
    }
    return x;
  };
  uint32_t causes = 0;
  for (int lane = 0; lane < 32; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    uint16_t x = source(a[lane], RJ_ARITH_ALU_ABS_A, RJ_ARITH_ALU_NEG_A);
    uint16_t y = source(b[lane], RJ_ARITH_ALU_ABS_B, RJ_ARITH_ALU_NEG_B);
    if constexpr (Op == Binary::Sub) {
      y ^= F::kSign;
    }
    if constexpr (Op == Binary::Subrev) {
      x ^= F::kSign;
    }
    const unsigned mx = x & F::kMagnitude, my = y & F::kMagnitude;
    if (mx >= F::kInfinity || my >= F::kInfinity) {
      const bool snan =
          (mx > F::kInfinity && !(x & F::kQuiet)) || (my > F::kInfinity && !(y & F::kQuiet));
      const bool invalid = Op == Binary::Mul
                               ? (mx == 0 && my == F::kInfinity) || (my == 0 && mx == F::kInfinity)
                               : mx == F::kInfinity && my == F::kInfinity && ((x ^ y) & F::kSign);
      if (snan || invalid) {
        causes |= RJ_ARITH_EXCEPTION_INVALID;
      }
      continue;
    }
    if ((mx && mx < F::kMinNormal) || (my && my < F::kMinNormal)) {
      causes |= RJ_ARITH_EXCEPTION_INPUT_DENORM;
    }
    const double lhs = f16_to_float(x), rhs = f16_to_float(y);
    const double exact = Op == Binary::Mul ? lhs * rhs : lhs + rhs;
    uint16_t arithmetic = half_fma_narrow(bit_cast<uint64_t>(exact), false, rounding);
    if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
        (arithmetic & F::kMagnitude) < F::kMinNormal) {
      arithmetic &= F::kSign;
    }
    causes |= gfx10_half_output_causes(exact, arithmetic, flags & RJ_ARITH_FP16_OVFL, omod);
    if constexpr (Gfx1010) {
      // Preserve record_gfx10_add_f16_exceptions, including its encoded tininess test.
      if ((arithmetic & F::kMagnitude) == F::kInfinity || __builtin_fabs(exact) >= 65536.0) {
        causes |= RJ_ARITH_EXCEPTION_OVERFLOW;
      }
      if (double(f16_to_float(arithmetic)) != exact) {
        causes |= RJ_ARITH_EXCEPTION_INEXACT;
        if (exact != 0 && __builtin_fabs(exact) < 0x1p-14) {
          causes |= RJ_ARITH_EXCEPTION_UNDERFLOW;
        }
      }
    } else {
      // Preserve gfx10_half_binary_causes: unbounded-exponent rounding decides overflow
      // and underflow independently of saturation, output flushing and CLAMP.
      const auto bits = bit_cast<uint64_t>(exact);
      const auto rounded = cm::conversion::round_finite<cm::fp_format::F64, F>(bits, policy);
      if (rounded >= F::kInfinity) {
        causes |= RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT;
      } else if (__builtin_fabs(exact) != f16_to_float(uint16_t(rounded))) {
        causes |= RJ_ARITH_EXCEPTION_INEXACT;
        if (__builtin_fabs(exact) < 0x1p-14 && cm::conversion::round_finite<cm::fp_format::F64, F>(
                                                   bits, {policy.direction, true}) == 0) {
          causes |= RJ_ARITH_EXCEPTION_UNDERFLOW;
        }
      }
    }
  }
  return causes;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
