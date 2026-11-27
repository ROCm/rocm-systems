// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited footprint precision stages, shared by GET_LOD and the sampler's
// arithmetic helpers. Definitions are isolated in each compiled CPU domain.
#pragma once
#include "common/isa.h"
#include "rj_arith/detail/image_footprint.h"
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::image_lod {
// Select the larger value, retaining A on ties.
template <typename T> constexpr T maximum(T a, T b) { return a < b ? b : a; }
// Select the smaller value, retaining A on ties.
template <typename T> constexpr T minimum(T a, T b) { return b < a ? b : a; }
// Bound a value to the supplied ordered interval.
template <typename T> constexpr T clamp(T value, T low, T high) {
  return value < low ? low : high < value ? high : value;
}
// Count significant bits, including zero for a zero word.
constexpr int bit_width(uint32_t value) { return value ? 32 - __builtin_clz(value) : 0; }
// Extract a frexp exponent directly for normal values; retain the library's
// handling of subnormal inputs outside the usual FP32-coordinate domain.
inline int frexp_exponent(double value) {
  const unsigned exponent = unsigned((__builtin_bit_cast(uint64_t, value) >> 52) & 2047);
  if (exponent && exponent != 2047) {
    return int(exponent) - 1022;
  }
  int result;
  __builtin_frexp(value, &result);
  return result;
}
// Multiplication by an exactly represented normal power of two rounds once,
// like ldexp. Extreme scales retain ldexp's gradual underflow/overflow behavior.
inline double scale(double value, int exponent) {
  if (exponent >= -1022 && exponent <= 1023) {
    return value * __builtin_bit_cast(double, uint64_t(exponent + 1023) << 52);
  }
  return __builtin_ldexp(value, exponent);
}
template <typename T, int Count> struct Array {
  T values[Count];
  // Access an element within the compile-time extent.
  constexpr T &operator[](int i) { return values[i]; }
  // Access an element without modifying it.
  constexpr const T &operator[](int i) const { return values[i]; }
  // Return the fixed extent.
  constexpr int size() const { return Count; }
};
// The five-projection fixed-point norm used by the inherited sampler model.
inline uint32_t polygon_norm(int32_t x, int32_t y) {
  const uint32_t a = maximum(__builtin_abs(x), __builtin_abs(y)),
                 b = minimum(__builtin_abs(x), __builtin_abs(y));
  return (maximum(maximum(256 * a + 16 * b, 250 * a + 56 * b),
                  maximum(238 * a + 96 * b, maximum(221 * a + 130 * b, 197 * a + 164 * b))) +
          128) >>
         8;
}
// Map an aligned integer norm to fixed8 LOD using the inherited eight log segments.
inline int32_t logarithm(uint32_t norm, int exponent) {
  if (!norm) {
    return -8192;
  }
  const int leading = bit_width(norm) - 1;
  const uint32_t denominator = 1U << leading, fraction = (norm - denominator) * 1024,
                 segment = fraction / (128 * denominator);
  constexpr uint32_t bases[] = {0, 175, 330, 471, 599, 717, 827, 929},
                     slopes[] = {176, 155, 142, 129, 119, 110, 102, 95};
  const uint32_t log =
      (bases[segment] * 128 * denominator +
       slopes[segment] * (fraction - segment * 128 * denominator) + 256 * denominator) /
      (512 * denominator);
  return (exponent - 11 + leading) * 256 + int32_t(log);
}
// Align and quantize the two screen derivatives, retaining axis norm carry bits.
inline rj_arith_image_footprint footprint_2d(double xu, double xv, double yu, double yv,
                                             uint32_t width, uint32_t height,
                                             const int32_t *rounding_directions) {
  const Array<double, 4> gradients{xu, xv, yu, yv};
  const Array<double, 2> largest{maximum(__builtin_fabs(xu), __builtin_fabs(yu)),
                                 maximum(__builtin_fabs(xv), __builtin_fabs(yv))};
  if (largest[0] == 0 && largest[1] == 0) {
    return {{}, {}, {-8192, -16384}};
  }
  const Array<uint32_t, 2> extents{width, height};
  Array<int, 2> exponents{};
  Array<double, 4> scaled{};
  for (int coordinate = 0; coordinate < 2; ++coordinate) {
    if (largest[coordinate] == 0) {
      continue;
    }
    const int coordinate_exponent = frexp_exponent(largest[coordinate]);
    const int extent_exponent = bit_width(extents[coordinate]) - 1;
    // The extent multiplier retains ten fractional bits, without rounding.
    const double extent_mantissa =
        __builtin_floor(scale(double(extents[coordinate]), 10 - extent_exponent)) / 1024;
    exponents[coordinate] = coordinate_exponent + extent_exponent;
    for (int i = coordinate; i < 4; i += 2) {
      const double mantissa = scale(__builtin_fabs(gradients[i]), 11 - coordinate_exponent);
      const bool toward_zero =
          rounding_directions[i] != 0 &&
          (rounding_directions[i] < 0) != bool(__builtin_signbit(gradients[i]));
      const double significand = minimum(2047.0, toward_zero ? __builtin_ceil(mantissa - 0.5)
                                                             : __builtin_floor(mantissa + 0.5));
      // Multiplication rounds signed ties upward after input conversion.
      scaled[i] =
          __builtin_floor(__builtin_copysign(significand, gradients[i]) * extent_mantissa + 0.5);
    }
  }
  const int exponent = largest[0] == 0   ? exponents[1]
                       : largest[1] == 0 ? exponents[0]
                                         : maximum(exponents[0], exponents[1]);
  Array<int32_t, 4> fixed{};
  for (int i = 0; i < gradients.size(); ++i) {
    fixed[i] =
        static_cast<int32_t>(__builtin_floor(scale(scaled[i], exponents[i % 2] - exponent) + 0.5));
  }
  const uint32_t first_norm = polygon_norm(fixed[0] + fixed[3], fixed[1] - fixed[2]);
  const uint32_t second_norm = polygon_norm(fixed[0] - fixed[3], fixed[1] + fixed[2]);
  const uint32_t major = (first_norm + second_norm) >> 1;
  // Round the signed difference before taking its magnitude. Negative odd
  // differences therefore produce a larger minor axis than positive ones.
  const uint32_t minor = __builtin_abs((int32_t(first_norm) - int32_t(second_norm)) >> 1);
  return {{fixed[0], fixed[1], 0, fixed[2], fixed[3], 0},
          {first_norm, second_norm},
          {logarithm(major, exponent), minor ? logarithm(minor, exponent) : -16384}};
}

// Quantize volume derivatives before the nested XY/Z norms.
inline rj_arith_image_footprint footprint_3d(const rj_arith_image_gradients &input,
                                             const uint32_t *extents) {
  const auto &screen_dx = input.dx;
  const auto &screen_dy = input.dy;
  constexpr int dimensions = 3;
  Array<double, 6> gradients{screen_dx[0], screen_dx[1], screen_dx[2],
                             screen_dy[0], screen_dy[1], screen_dy[2]};
  Array<double, 3> largest_gradient{};
  for (int coordinate = 0; coordinate < dimensions; ++coordinate) {
    for (int gradient_index = coordinate; gradient_index < 6; gradient_index += 3) {
      if (__builtin_isnan(gradients[gradient_index])) {
        gradients[gradient_index] = 0;
      } else if (__builtin_isinf(gradients[gradient_index])) {
        gradients[gradient_index] =
            __builtin_copysign(double(0x1.fffffep127f), gradients[gradient_index]);
      }
    }
    largest_gradient[coordinate] =
        maximum(__builtin_fabs(gradients[coordinate]), __builtin_fabs(gradients[coordinate + 3]));
  }
  if (largest_gradient[0] == 0 && largest_gradient[1] == 0 && largest_gradient[2] == 0) {
    return {{}, {}, {-8192, -16384}};
  }
  Array<int, 3> exponents{};
  Array<double, 6> scaled{};
  int exponent = INT32_MIN;
  for (int coordinate = 0; coordinate < dimensions; ++coordinate) {
    if (largest_gradient[coordinate] == 0) {
      continue;
    }
    const int coordinate_exponent = frexp_exponent(largest_gradient[coordinate]);
    const int extent_exponent = bit_width(extents[coordinate]) - 1;
    const double extent_mantissa =
        __builtin_floor(scale(double(extents[coordinate]), 10 - extent_exponent)) / 1024;
    exponents[coordinate] = coordinate_exponent + extent_exponent;
    exponent = maximum(exponent, exponents[coordinate]);
    for (int gradient_index = coordinate; gradient_index < 6; gradient_index += 3) {
      const double significand =
          minimum(2047.0, __builtin_floor(scale(__builtin_fabs(gradients[gradient_index]),
                                                11 - coordinate_exponent) +
                                          0.5));
      scaled[gradient_index] = __builtin_floor(
          __builtin_copysign(significand, gradients[gradient_index]) * extent_mantissa + 0.5);
    }
  }
  Array<int32_t, 6> fixed{};
  for (int coordinate = 0; coordinate < dimensions; ++coordinate) {
    if (largest_gradient[coordinate] == 0) {
      continue;
    }
    for (int gradient_index = coordinate; gradient_index < 6; gradient_index += 3) {
      fixed[gradient_index] = static_cast<int32_t>(
          __builtin_floor(scale(scaled[gradient_index], exponents[coordinate] - exponent) + 0.5));
    }
  }
  const uint32_t first = polygon_norm(polygon_norm(fixed[0], fixed[1]), fixed[2]);
  const uint32_t second = polygon_norm(polygon_norm(fixed[3], fixed[4]), fixed[5]);
  return {{fixed[0], fixed[1], fixed[2], fixed[3], fixed[4], fixed[5]},
          {first, second},
          {logarithm(maximum(first, second), exponent), -16384}};
}
// Sanitize derivatives without erasing a NaN before quad subtraction.
inline double finite_gradient(double value) {
  return __builtin_isnan(value)   ? 0.0
         : __builtin_isinf(value) ? __builtin_copysign(double(0x1.fffffep127f), value)
                                  : value;
}
// Compute a 1D/2D/3D footprint; normalized extents have already been resolved.
inline rj_arith_image_footprint footprint(const rj_arith_image_gradients &input,
                                          const uint32_t *extents, int dimensions) {
  if (dimensions == 3) {
    return footprint_3d(input, extents);
  }
  return footprint_2d(
      finite_gradient(input.dx[0]), dimensions == 1 ? 0 : finite_gradient(input.dx[1]),
      finite_gradient(input.dy[0]), dimensions == 1 ? 0 : finite_gradient(input.dy[1]), extents[0],
      dimensions == 1 ? 1 : extents[1], input.rounding);
}
// Remap a fractional mip according to the decoded sampler performance controls.
inline uint32_t mip_fraction(uint32_t fraction, uint32_t perf_mip, uint32_t perf_mod) {
  constexpr uint32_t scales[] = {0, 2, 5, 7, 9, 11, 14, 16};
  constexpr int32_t gains[] = {16, 17, 18, 20, 24, 32, 34, 36, 40, 48, 64, 72, 80, 96, 128, 192};
  const uint32_t index = perf_mip * scales[perf_mod] >> 4;
  return uint32_t(clamp(2048 + (int32_t(fraction) - 128) * gains[index], 0, 4096)) / 16;
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::image_lod
