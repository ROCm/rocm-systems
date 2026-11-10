// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Guest FP32 multiplication exception classification from operands and result.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/bits.h"
#include "fp/fma_integer.h"
#include "fp/fma_scalar.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// gfx1201 MUL status for nearest-even arithmetic with preserved denormals and
// no output modifiers. Finite FP32 products fit exactly in the FP64 intermediate.
inline uint32_t mul_exceptions_f32(uint32_t a, uint32_t b) {
  a &= 0x7fffffffU;
  b &= 0x7fffffffU;
  uint32_t causes =
      ((a && a < 0x00800000U) || (b && b < 0x00800000U)) ? RJ_ARITH_EXCEPTION_INPUT_DENORM : 0;
  const bool signaling =
      (a > 0x7f800000U && !(a & 0x00400000U)) || (b > 0x7f800000U && !(b & 0x00400000U));
  // NaNs suppress input-denormal reporting; signaling NaNs still raise INVALID.
  if (a > 0x7f800000U || b > 0x7f800000U) {
    return signaling ? RJ_ARITH_EXCEPTION_INVALID : 0;
  }
  if (signaling || (!a && b == 0x7f800000U) || (!b && a == 0x7f800000U)) {
    causes |= RJ_ARITH_EXCEPTION_INVALID;
  }
  if (a >= 0x7f800000U || b >= 0x7f800000U) {
    return causes;
  }
  uint64_t exact = fma_finite_round_odd(a, b, 0);
  uint32_t rounded = fma_narrow_f32(exact, false);
  const bool inexact =
      rounded == 0x7f800000U || fma_finite_round_odd(rounded, 0x3f800000U, 0) != exact;
  if (rounded == 0x7f800000U) {
    causes |= RJ_ARITH_EXCEPTION_OVERFLOW;
  }
  if (inexact) {
    causes |= RJ_ARITH_EXCEPTION_INEXACT;
    // Test tininess after rounding the significand with an unbounded exponent.
    // At the midpoint 2^-126 - 2^-151, ties-to-even rounds up to minimum normal.
    if (exact && exact < 0x380ffffff0000000ULL) {
      causes |= RJ_ARITH_EXCEPTION_UNDERFLOW;
    }
  }
  return causes;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
