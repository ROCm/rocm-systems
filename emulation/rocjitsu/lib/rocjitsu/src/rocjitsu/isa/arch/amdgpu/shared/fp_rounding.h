// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Explicit IEEE rounding for the retained graphics arithmetic. Host arithmetic
// uses the default environment; guest rounding is applied to the result bits.
// TwoSum retains small addends and round-to-odd prevents double rounding.
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu::amdgpu::fp_mode::detail {

struct ExactF64Sum {
  double value;
  double error;
};

/// @brief Add two finite F64 values while retaining the exact rounding residual.
inline ExactF64Sum add_exact(double lhs, double rhs) {
  const double value = lhs + rhs;
  const double rhs_virtual = value - lhs;
  const double error = (lhs - (value - rhs_virtual)) + (rhs - rhs_virtual);
  return {value, error};
}

// Retain the direction of an exact addition's residual in its low bit.
inline double round_to_odd(ExactF64Sum sum) {
  if (sum.error == 0.0 || !std::isfinite(sum.value) || (std::bit_cast<uint64_t>(sum.value) & 1u)) {
    return sum.value;
  }
  return std::nextafter(sum.value,
                        std::copysign(std::numeric_limits<double>::infinity(), sum.error));
}

// Narrow a finite FP64 round-to-odd value using the MODE rounding encoding.
// This is the scalar narrowing rule already used by rj-arith's F32 FMA.
inline uint32_t narrow_f32_bits(uint64_t bits, bool flush_output, unsigned rounding) {
  uint32_t sign = uint32_t(bits >> 32) & 0x80000000;
  uint64_t magnitude = bits & 0x7fffffffffffffffULL;
  if (!magnitude) {
    return sign;
  }
  const bool away = rounding == (sign ? 2u : 1u);
  const uint32_t overflow = sign | ((rounding == 0 || away) ? 0x7f800000 : 0x7f7fffff);
  int exponent = int(magnitude >> 52) - 1023;
  if (exponent > 127) {
    return overflow;
  }
  if (flush_output && exponent < -127) {
    return sign;
  }
  if (exponent < -150) {
    return sign | unsigned(away);
  }
  int shift = 29 + (!flush_output && exponent < -126 ? -126 - exponent : 0);
  uint64_t significand = (magnitude & 0xfffffffffffffULL) | (1ULL << 52);
  uint32_t rounded = uint32_t(significand >> shift);
  uint64_t unit = 1ULL << shift, remainder = significand & (unit - 1);
  rounded += rounding == 0 ? remainder > unit / 2 || (remainder == unit / 2 && (rounded & 1))
                           : away && remainder != 0;
  if (!flush_output && exponent < -126) {
    return sign | rounded;
  }
  if (rounded == (1U << 24)) {
    rounded >>= 1;
    ++exponent;
  }
  if (exponent > 127) {
    return overflow;
  }
  // With output flushing, an unpromoted tiny significand becomes signed zero.
  if (exponent < -126) {
    return sign;
  }
  return sign | (uint32_t(exponent + 127) << 23) | (rounded & 0x7fffff);
}

// Round a finite exact sum; cancellation is negative only for downward rounding.
inline float sum_f32(ExactF64Sum sum, uint32_t rounding, bool flush_output) {
  if (sum.value == 0 && sum.error == 0) {
    return std::bit_cast<float>((rounding == 2) ? 0x80000000u : 0u);
  }
  return std::bit_cast<float>(
      narrow_f32_bits(std::bit_cast<uint64_t>(round_to_odd(sum)), flush_output, rounding));
}

// Add at F32 precision with explicit rounding, preserving matching signed zeros.
inline float add_f32(float a, float b, uint32_t rounding) {
  if (!std::isfinite(a) || !std::isfinite(b)) {
    return a + b;
  }
  const auto sum = add_exact(double(a), double(b));
  if (a == 0 && b == 0 && std::signbit(a) == std::signbit(b)) {
    return a;
  }
  return sum_f32(sum, rounding, false);
}

// Subtract without changing a NaN subtrahend's sign through source negation.
inline float subtract_f32(float a, float b, uint32_t rounding) {
  return std::isfinite(a) && std::isfinite(b) ? add_f32(a, -b, rounding) : a - b;
}

// Multiply at F32 precision; every finite F32 product is exact in FP64.
inline float multiply_f32(float a, float b, uint32_t rounding) {
  if (!std::isfinite(a) || !std::isfinite(b)) {
    return a * b;
  }
  const double product = double(a) * double(b);
  return std::bit_cast<float>(narrow_f32_bits(std::bit_cast<uint64_t>(product), false, rounding));
}

// Evaluate a finite F32 FMA with explicit rounding and optional output flushing.
inline float fma_finite_f32(float a, float b, float c, uint32_t rounding, bool flush_output) {
  const auto sum = add_exact(double(a) * double(b), double(c));
  const bool zero_product = a == 0 || b == 0;
  if (zero_product && c == 0 && (std::signbit(a) != std::signbit(b)) == std::signbit(c)) {
    return c;
  }
  return sum_f32(sum, rounding, flush_output);
}

} // namespace rocjitsu::amdgpu::fp_mode::detail
