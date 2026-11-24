// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar FP exception classification, inherited from Rocjitsu shared/fp_status.h with the
// hardware overflow/tininess rule fitted to gfx1201 captures.
// Numerical causes belong here; committing guest registers and traps belongs to the VM.

#pragma once

#include "common/float_encoding.h"
#include "common/validation.h"
#include "fp/fma_integer.h"
#include "fp/rounding_status.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Decode one scalar FP16/FP32 encoding without changing host FP controls.
template <bool Half> inline double scalar_fp_value(uint32_t bits) {
  return Half ? double(f16_to_float(uint16_t(bits))) : double(as_float(bits));
}

// Preserve scalar input classification: flushing suppresses denormal reporting.
template <bool Half> inline uint32_t scalar_input_causes(uint32_t bits, uint64_t flags) {
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

// Classify a wider, round-to-odd intermediate against the scalar destination with
// Rocjitsu's inherited overflow thresholds and stored-result underflow rule. Only gfx1150
// keeps it: the shared rule is fitted to gfx1201 captures and gfx1150 has none.
template <bool Half>
inline uint32_t scalar_result_causes(double exact, uint32_t result, unsigned rounding) {
  if (exact == scalar_fp_value<Half>(result)) {
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

// Classify the scalar FMA model, also used for ADD/SUB/MUL lowering, with the shared
// hardware overflow/tininess rule of fp/rounding_status.h, or the inherited
// stored-result rule for gfx1150.
// Preserve the inherited suppression of input-denormal causes for NaN results.
template <bool Half, bool Inherited = false>
inline uint32_t scalar_fma_causes(uint64_t flags, uint32_t a, uint32_t b, uint32_t c,
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
    causes |= scalar_input_causes<Half>(input, flags);
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
    const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
    if constexpr (Inherited) {
      // The inherited model uses an exact FP64 product and TwoSum residual.
      // FP16/FP32 operands cannot overflow or underflow FP64 here. With default
      // host rounding, moving an even sum one encoding toward the residual
      // retains its sticky information before the destination is classified.
      const double product = scalar_fp_value<Half>(a) * scalar_fp_value<Half>(b);
      const double addend = scalar_fp_value<Half>(c), sum = product + addend;
      const double part = sum - product;
      const double residual = (product - (sum - part)) + (addend - part);
      uint64_t bits = double_bits(sum);
      if (residual != 0 && !(bits & 1)) {
        bits = (sum > 0) == (residual > 0) ? bits + 1 : bits - 1;
      }
      causes |= scalar_result_causes<Half>(as_double(bits), result, rounding);
    } else {
      // Exact round-to-odd FP64 intermediate; FP16 operands widen exactly to FP32.
      auto widen = [](uint32_t bits) {
        return Half ? as_bits(f16_to_float(uint16_t(bits))) : bits;
      };
      const uint64_t exact = fma_finite_round_odd(widen(a), widen(b), widen(c));
      if (exact & 0x7fffffffffffffffULL) {
        causes |= rounding_causes<Half>(exact, result, rounding);
      }
    }
  }
  if ((result & magnitude) > infinity) {
    causes &= ~RJ_ARITH_EXCEPTION_INPUT_DENORM;
  }
  return causes;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
