// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Bitwise rounding of FP16/FP32 to integral values under guest rounding rules.
// Internal helpers shared by instruction implementations; not part of the public C API.
// Extends rocjitsu util::rndne_scalar's integer algorithm to directed rounding.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class IntegralRound { Ceil, Floor, Trunc, Nearest };

// Round raw FP16/FP32 bits to an integral value. Preserve signed zero, quiet
// signaling NaNs, and optionally flush input subnormals. Results do not depend
// on host FP rounding or denormal controls.
template <IntegralRound Op, bool Half>
inline uint32_t round_integral_bits(uint32_t a, bool flush_input) {
  constexpr int fraction = Half ? 10 : 23, bias = Half ? 15 : 127;
  constexpr uint32_t sign_bit = Half ? 0x8000 : 0x80000000, inf = Half ? 0x7c00 : 0x7f800000;
  if constexpr (Half) {
    a &= 65535;
  }
  if (flush_input && !(a & inf)) {
    a &= sign_bit;
  }
  uint32_t magnitude = a & (sign_bit - 1), sign = a & sign_bit, result;
  int exponent = magnitude >> fraction;
  if (magnitude >= inf) {
    result = a | (magnitude > inf ? (1u << (fraction - 1)) : 0);
  } else if (exponent >= bias + fraction) {
    result = a;
  } else if (exponent < bias) {
    bool up = false;
    if constexpr (Op == IntegralRound::Ceil) {
      up = !sign && magnitude;
    }
    if constexpr (Op == IntegralRound::Floor) {
      up = sign && magnitude;
    }
    if constexpr (Op == IntegralRound::Nearest) {
      up = magnitude > ((bias - 1) << fraction);
    }
    result = sign | (up ? bias << fraction : 0);
  } else {
    uint32_t unit = 1u << (bias + fraction - exponent), tail = magnitude & (unit - 1),
             rounded = magnitude & ~(unit - 1);
    bool up = false;
    if constexpr (Op == IntegralRound::Ceil) {
      up = !sign && tail;
    }
    if constexpr (Op == IntegralRound::Floor) {
      up = sign && tail;
    }
    if constexpr (Op == IntegralRound::Nearest) {
      up = tail > unit / 2 || (tail == unit / 2 && (rounded & unit));
    }
    result = sign | (rounded + (up ? unit : 0));
  }
  return result;
}

} // namespace rocjitsu::arith
