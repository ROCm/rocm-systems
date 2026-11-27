// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited compensated rounding for gfx10 interpolation and scalar BVH arithmetic.
// The host uses its default FP controls; guest rounding never changes those controls.
#pragma once

#include "common/bits.h"
#include "common/float_encoding.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::graphics_rounding {

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
  if (sum.error == 0.0 || !__builtin_isfinite(sum.value) || (bit_cast<uint64_t>(sum.value) & 1u)) {
    return sum.value;
  }
  return __builtin_nextafter(sum.value, __builtin_copysign(__builtin_inf(), sum.error));
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
    return bit_cast<float>((rounding == 2) ? 0x80000000u : 0u);
  }
  return bit_cast<float>(
      narrow_f32_bits(bit_cast<uint64_t>(round_to_odd(sum)), flush_output, rounding));
}

// Add at F32 precision with explicit rounding, preserving matching signed zeros.
inline float add_f32(float a, float b, uint32_t rounding) {
  if (!__builtin_isfinite(a) || !__builtin_isfinite(b)) {
    return a + b;
  }
  const auto sum = add_exact(double(a), double(b));
  if (a == 0 && b == 0 && bool(__builtin_signbit(a)) == bool(__builtin_signbit(b))) {
    return a;
  }
  return sum_f32(sum, rounding, false);
}

// Subtract without changing a NaN subtrahend's sign through source negation.
inline float subtract_f32(float a, float b, uint32_t rounding) {
  return __builtin_isfinite(a) && __builtin_isfinite(b) ? add_f32(a, -b, rounding) : a - b;
}

// Multiply at F32 precision; every finite F32 product is exact in FP64.
inline float multiply_f32(float a, float b, uint32_t rounding) {
  if (!__builtin_isfinite(a) || !__builtin_isfinite(b)) {
    return a * b;
  }
  const double product = double(a) * double(b);
  return bit_cast<float>(narrow_f32_bits(bit_cast<uint64_t>(product), false, rounding));
}

// Evaluate a finite F32 FMA with explicit rounding and optional output flushing.
inline float fma_finite_f32(float a, float b, float c, uint32_t rounding, bool flush_output) {
  const auto sum = add_exact(double(a) * double(b), double(c));
  const bool zero_product = a == 0 || b == 0;
  if (zero_product && c == 0 &&
      (bool(__builtin_signbit(a)) != bool(__builtin_signbit(b))) == bool(__builtin_signbit(c))) {
    return c;
  }
  return sum_f32(sum, rounding, flush_output);
}

namespace interp_half {

inline uint16_t next_up_f16(uint16_t value) {
  if ((value & 0x7fffu) > 0x7c00u || value == 0x7c00u) {
    return value;
  }
  if (value == 0xfc00u) {
    return 0xfbffu;
  }
  if ((value & 0x7fffu) == 0) {
    return 0x0001u;
  }
  return static_cast<uint16_t>((value & 0x8000u) != 0 ? value - 1u : value + 1u);
}

inline uint16_t next_down_f16(uint16_t value) {
  if ((value & 0x7fffu) > 0x7c00u || value == 0xfc00u) {
    return value;
  }
  if (value == 0x7c00u) {
    return 0x7bffu;
  }
  if ((value & 0x7fffu) == 0) {
    return 0x8001u;
  }
  return static_cast<uint16_t>((value & 0x8000u) != 0 ? value + 1u : value - 1u);
}

// Round an interpolation result to F16 with guest rounding and overflow policy.
inline uint16_t round_f64_to_f16(double value, uint32_t round_mode, bool fp16_ovfl) {
  if (__builtin_isnan(value) || __builtin_isinf(value)) {
    return float_to_f16(static_cast<float>(value));
  }

  constexpr double MAX_F16 = 65504.0;
  constexpr double RNE_OVERFLOW_THRESHOLD = 65520.0;
  if (__builtin_fabs(value) > MAX_F16) {
    const bool negative = bool(__builtin_signbit(value));
    bool to_infinity = false;
    switch (round_mode & 3u) {
    case 0:
      to_infinity = __builtin_fabs(value) >= RNE_OVERFLOW_THRESHOLD;
      break;
    case 1:
      to_infinity = !negative;
      break;
    case 2:
      to_infinity = negative;
      break;
    default:
      break;
    }
    if (to_infinity && !fp16_ovfl) {
      return negative ? 0xfc00u : 0x7c00u;
    }
    return negative ? 0xfbffu : 0x7bffu;
  }

  const uint16_t candidate = float_to_f16(static_cast<float>(value));
  const double candidate_value = f16_to_float(candidate);
  if (candidate_value == value) {
    return candidate;
  }

  const uint16_t lower = candidate_value < value ? candidate : next_down_f16(candidate);
  const uint16_t upper = candidate_value > value ? candidate : next_up_f16(candidate);

  switch (round_mode & 3u) {
  case 0: {
    const double lower_distance = value - f16_to_float(lower);
    const double upper_distance = f16_to_float(upper) - value;
    if (lower_distance < upper_distance) {
      return lower;
    }
    if (upper_distance < lower_distance) {
      return upper;
    }
    return (lower & 1u) == 0 ? lower : upper;
  }
  case 1:
    return upper;
  case 2:
    return lower;
  case 3:
    return value < 0.0 ? upper : lower;
  default:
    return candidate;
  }
}

} // namespace interp_half

// Clamp the interpolation result, then round directly from FP64 to F16.
inline uint16_t round_f16_result(double value, uint32_t round_mode, bool clamp, bool fp16_ovfl,
                                 bool clamp_nan_to_zero) {
  if (clamp && (clamp_nan_to_zero || !__builtin_isnan(value))) {
    if (__builtin_isnan(value) || value <= 0.0) {
      value = 0.0;
    } else if (value > 1.0) {
      value = 1.0;
    }
  }
  return interp_half::round_f64_to_f16(value, round_mode, fp16_ovfl);
}

/// Round an F32-source FMA directly to F16 in a clean nearest environment.
inline uint16_t fma_f32_to_f16_nearest_environment(float a, float b, float c, uint32_t round_mode,
                                                   bool clamp, bool fp16_ovfl,
                                                   bool clamp_nan_to_zero) {
  double value;
  if (__builtin_isfinite(a) && __builtin_isfinite(b) && __builtin_isfinite(c)) {
    const auto exact = add_exact(double(a) * double(b), double(c));
    value = round_to_odd(exact);
    if (exact.value == 0 && exact.error == 0) {
      // Finite F32 products are exact, normal F64 values or signed zero.
      // Matching signed zeros keep their sign; cancellation and opposite
      // signed zeros are negative only when rounding toward negative infinity.
      const uint32_t product_sign = (bit_cast<uint32_t>(a) ^ bit_cast<uint32_t>(b)) >> 31;
      const uint32_t addend_bits = bit_cast<uint32_t>(c);
      const bool matching_zeros =
          (addend_bits & 0x7fffffffu) == 0 && product_sign == (addend_bits >> 31);
      const bool negative = matching_zeros ? product_sign != 0 : (round_mode & 3u) == 2u;
      value = bit_cast<double>(uint64_t{negative} << 63);
    }
  } else {
    const uint32_t a_abs = bit_cast<uint32_t>(a) & 0x7fffffffu;
    const uint32_t b_abs = bit_cast<uint32_t>(b) & 0x7fffffffu;
    // The invalid product wins even when the addend supplies a NaN payload.
    const bool invalid_product =
        (a_abs == 0 && b_abs == 0x7f800000u) || (b_abs == 0 && a_abs == 0x7f800000u);
    value = invalid_product ? bit_cast<double>(uint64_t{0xfff8000000000000})
                            : __builtin_fma(double(a), double(b), double(c));
  }
  return round_f16_result(value, round_mode, clamp, fp16_ovfl, clamp_nan_to_zero);
}

// Apply interpolation's input flushing, NaN policy and guest rounding without
// changing host controls. Finite arithmetic uses FP64 with a retained residual.
inline float fma_f32(float a, float b, float c, uint32_t round_mode, uint32_t denorm_mode,
                     bool ieee_mode) {
  auto input_bits = [denorm_mode](float value) {
    const uint32_t bits = bit_cast<uint32_t>(value);
    return !(denorm_mode & 1u) && (bits & 0x7f800000u) == 0 ? bits & 0x80000000u : bits;
  };
  const uint32_t a_bits = input_bits(a), b_bits = input_bits(b), c_bits = input_bits(c);
  const uint32_t a_abs = a_bits & 0x7fffffffu, b_abs = b_bits & 0x7fffffffu;
  if ((a_abs == 0 && b_abs == 0x7f800000u) || (b_abs == 0 && a_abs == 0x7f800000u)) {
    return bit_cast<float>(0xffc00000u);
  }
  const uint32_t inputs[] = {a_bits, b_bits, c_bits};
  for (const uint32_t bits : inputs) {
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
      return bit_cast<float>(bits | (ieee_mode ? 0x00400000u : 0u));
    }
  }
  a = bit_cast<float>(a_bits);
  b = bit_cast<float>(b_bits);
  c = bit_cast<float>(c_bits);
  if (__builtin_isfinite(a) && __builtin_isfinite(b) && __builtin_isfinite(c)) {
    return fma_finite_f32(a, b, c, round_mode & 3u, !(denorm_mode & 2u));
  }
  const float result = __builtin_fma(a, b, c);
  return __builtin_isnan(result) ? bit_cast<float>(0xffc00000u) : result;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::graphics_rounding
