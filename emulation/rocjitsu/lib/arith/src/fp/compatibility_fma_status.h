// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited FP16/FP32 FMA exception classification from Rocjitsu shared/fp_status.h.
// Numerical causes belong here; committing guest registers and traps belongs to the VM.

#pragma once

#include "common/float_encoding.h"
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Decode one FP16/FP32 encoding without changing host FP controls.
template <bool Half> inline double compatibility_fp_value(uint32_t bits) {
  return Half ? double(f16_to_float(uint16_t(bits))) : double(as_float(bits));
}

// Preserve input classification: flushing suppresses denormal reporting.
template <bool Half> inline uint32_t compatibility_input_causes(uint32_t bits, uint64_t flags) {
  constexpr uint32_t magnitude = Half ? 0x7fff : 0x7fffffff;
  constexpr uint32_t infinity = Half ? 0x7c00 : 0x7f800000;
  constexpr uint32_t quiet = Half ? 0x200 : 0x400000;
  constexpr uint32_t normal = Half ? 0x400 : 0x800000;
  bits &= magnitude;
  if (bits > infinity && !(bits & quiet)) {
    return RJ_ARITH_EXCEPTION_INVALID;
  }
  return !(flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && bits && bits < normal
             ? RJ_ARITH_EXCEPTION_INPUT_DENORM
             : 0;
}

// Classify a wider, round-to-odd intermediate against the destination.
// This retains Rocjitsu's overflow thresholds and after-rounding underflow rule.
template <bool Half>
inline uint32_t compatibility_result_causes(double exact, uint32_t result, unsigned rounding) {
  if (exact == compatibility_fp_value<Half>(result)) {
    return 0;
  }
  constexpr double limit = Half ? 0x1p16 : 0x1p128;
  constexpr double ulp = Half ? 0x1p5 : 0x1p104;
  constexpr uint32_t magnitude_mask = Half ? 0x7fff : 0x7fffffff;
  constexpr uint32_t normal = Half ? 0x400 : 0x800000;
  const double magnitude = __builtin_fabs(exact);
  const bool outward = (rounding == 1 && exact > 0) || (rounding == 2 && exact < 0);
  const bool overflow = rounding == 0 ? magnitude >= limit - ulp / 2
                        : outward     ? magnitude > limit - ulp
                                      : magnitude >= limit;
  return RJ_ARITH_EXCEPTION_INEXACT |
         (overflow                             ? RJ_ARITH_EXCEPTION_OVERFLOW
          : (result & magnitude_mask) < normal ? RJ_ARITH_EXCEPTION_UNDERFLOW
                                               : 0);
}

// Classify inherited FMA arithmetic, also used for ADD/SUB/MUL lowering.
template <bool Half>
inline uint32_t compatibility_fma_causes(uint64_t flags, uint32_t a, uint32_t b, uint32_t c,
                                         uint32_t result) {
  constexpr uint32_t sign = Half ? 0x8000 : 0x80000000;
  constexpr uint32_t magnitude = sign - 1;
  constexpr uint32_t infinity = Half ? 0x7c00 : 0x7f800000;
  constexpr uint32_t normal = Half ? 0x400 : 0x800000;
  uint32_t inputs[] = {a, b, c};
  uint32_t causes = 0;
  for (auto &input : inputs) {
    input &= sign | magnitude;
    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (input & magnitude) < normal) {
      input &= sign;
    }
    causes |= compatibility_input_causes<Half>(input, flags);
  }
  a = inputs[0];
  b = inputs[1];
  c = inputs[2];
  const uint32_t x = a & magnitude, y = b & magnitude, z = c & magnitude;
  if ((x == 0 && y == infinity) || (x == infinity && y == 0) ||
      ((x == infinity || y == infinity) && x <= infinity && y <= infinity && z == infinity &&
       ((a ^ b ^ c) & sign))) {
    causes |= RJ_ARITH_EXCEPTION_INVALID;
  }
  if (x < infinity && y < infinity && z < infinity) {
    // The inherited model uses an exact FP64 product and TwoSum residual.
    // FP16/FP32 operands cannot overflow or underflow FP64 here. With default
    // host rounding, moving an even sum one encoding toward the residual
    // retains its sticky information before the destination is classified.
    const double product = compatibility_fp_value<Half>(a) * compatibility_fp_value<Half>(b);
    const double addend = compatibility_fp_value<Half>(c), sum = product + addend;
    const double part = sum - product;
    const double residual = (product - (sum - part)) + (addend - part);
    uint64_t bits = double_bits(sum);
    if (residual != 0 && !(bits & 1)) {
      bits = (sum > 0) == (residual > 0) ? bits + 1 : bits - 1;
    }
    const double exact = as_double(bits);
    causes |=
        compatibility_result_causes<Half>(exact, result, (flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  }
  return causes;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
