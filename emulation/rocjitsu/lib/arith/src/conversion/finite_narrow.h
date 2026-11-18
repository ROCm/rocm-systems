// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited Rocjitsu FP4/FP6/BF6 rounding, including its stochastic tiny-value cutoff.
#pragma once
#include "common/isa.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Narrow FP32 bits to a finite-only format, adapted from util/data_types.h.
// NaNs become positive zero; infinities saturate. Magnitudes below half the
// smallest nonzero value flush to signed zero, including stochastic rounding.
template <int ExponentBits, int FractionBits, bool Stochastic>
inline uint32_t narrow_finite(uint32_t raw, uint32_t seed) {
  constexpr int bias = (1 << (ExponentBits - 1)) - 1;
  constexpr uint32_t sign_bit = 1U << (ExponentBits + FractionBits);
  constexpr uint32_t max_code = sign_bit - 1;
  constexpr uint32_t max_value = ((127 + (1 << ExponentBits) - 1 - bias) << 23) |
                                 (((1U << FractionBits) - 1) << (23 - FractionBits));
  constexpr uint32_t cutoff = (127 - bias - FractionBits) << 23;
  uint32_t magnitude = raw & 0x7fffffffU;
  uint32_t sign = (raw >> (31 - ExponentBits - FractionBits)) & sign_bit;
  if (magnitude > 0x7f800000U) {
    return 0;
  }
  if (magnitude > max_value) {
    return sign | max_code;
  }
  if (magnitude < cutoff) {
    return sign;
  }
  int exponent = int(magnitude >> 23) - 127 + bias;
  int shift = exponent <= 0 ? 24 - FractionBits - exponent : 23 - FractionBits;
  uint32_t significand = (raw & 0x7fffffU) | 0x800000U;
  uint32_t increment;
  if constexpr (Stochastic) {
    increment = seed >> (32 - shift);
  } else {
    increment = (1U << (shift - 1)) - 1 + ((significand >> shift) & 1);
  }
  uint32_t result = (significand + increment) >> shift;
  if (exponent > 0) {
    result += uint32_t(exponent - 1) << FractionBits;
  }
  return sign | (result < max_code ? result : max_code);
}

// Advance the architectural seed for the next packed result. This is
// the instruction's inherited recurrence, not a test-input random engine.
inline uint32_t narrow_next_seed(uint32_t seed) { return (seed << 1) ^ ((seed >> 31) ? 197U : 0U); }
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
