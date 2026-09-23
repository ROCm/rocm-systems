// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

namespace rocjitsu::amdgpu::raster {

inline float truncate_float(double value) {
  float result = static_cast<float>(value);
  if (std::abs(static_cast<double>(result)) > std::abs(value))
    result = std::bit_cast<float>(std::bit_cast<uint32_t>(result) - 1);
  return result;
}

// Raster setup uses a linear reciprocal seed followed by a fixed-point
// Newton step. These coefficients and rounding stages match every normalized
// FP32 mantissa captured on physical RDNA3 and RDNA4. Shader reciprocals use
// a different approximation.
inline float reciprocal(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude < 0x00800000u || magnitude >= 0x7f800000u)
    return 1.0f / value;
  struct Coefficient {
    uint16_t base;
    uint8_t slope;
  };
  static constexpr Coefficient coefficients[] = {
      {32767, 252}, {32262, 244}, {31774, 237}, {31299, 230}, {30839, 223}, {30392, 217},
      {29958, 210}, {29536, 205}, {29126, 199}, {28727, 194}, {28338, 188}, {27961, 183},
      {27593, 179}, {27234, 174}, {26885, 170}, {26545, 165}, {26213, 161}, {25889, 157},
      {25574, 154}, {25265, 150}, {24965, 146}, {24671, 143}, {24384, 140}, {24104, 136},
      {23830, 133}, {23562, 130}, {23300, 128}, {23044, 125}, {22794, 122}, {22549, 119},
      {22309, 117}, {22074, 114}, {21844, 112}, {21619, 110}, {21398, 108}, {21182, 105},
      {20970, 103}, {20762, 101}, {20559, 99},  {20359, 97},  {20163, 96},  {19971, 94},
      {19783, 92},  {19598, 90},  {19417, 89},  {19238, 87},  {19064, 85},  {18892, 84},
      {18723, 82},  {18557, 81},  {18395, 79},  {18235, 78},  {18077, 77},  {17923, 75},
      {17771, 74},  {17622, 73},  {17475, 72},  {17330, 71},  {17188, 69},  {17049, 68},
      {16911, 67},  {16776, 66},  {16643, 65},  {16512, 64},
  };
  const uint32_t mantissa = magnitude & 0x7fffffu;
  const Coefficient coefficient = coefficients[mantissa >> 17];
  // Keep seven fraction bits and fold discarded bits into a sticky bit.
  const uint32_t fraction = ((mantissa >> 9) & 0xfeu) | ((mantissa & 0x3ffu) != 0);
  const uint32_t seed =
      (uint32_t{coefficient.base} * 128 - coefficient.slope * fraction + 128) >> 8;
  const uint32_t product = (uint64_t{mantissa | 0x800000u} * seed) >> 11;
  const uint64_t refined = uint64_t{seed} * ((1u << 27) - product);
  const uint32_t normalized = 0x3e800000u + uint32_t((refined + 0x8000u) >> 16);
  const float result = std::ldexp(std::bit_cast<float>(normalized), 127 - int(magnitude >> 23));
  return std::copysign(result, value);
}

// Setup truncates the weighted numerator before scaling by reciprocal area.
// Weighting already-divided barycentric gradients changes the low bits.
inline float plane_gradient(double edge1, double edge2, double delta1, double delta2,
                            float inverse_area) {
  const float numerator = truncate_float(edge1 * delta1 + edge2 * delta2);
  return truncate_float(double(numerator) * inverse_area);
}

// Physical RDNA3/4 perspective interpolation rounds the product using the
// input exponents, then normalizes it with truncation. A carry into another
// exponent therefore discards a bit after rounding. Midpoints round away
// from zero, unlike an IEEE float multiplication.
inline float multiply_perspective(float a, float b) {
  if (a == 0 || b == 0 || !std::isfinite(a) || !std::isfinite(b))
    return a * b;
  const double product = double(a) * b;
  const double unit = std::ldexp(1.0, std::ilogb(a) + std::ilogb(b) - 23);
  const double rounded = std::floor(std::abs(product) / unit + 0.5) * unit;
  return truncate_float(std::copysign(rounded, product));
}

// Rasterizer quad offsets use a shared exponent and discard shifted-out bits
// before addition. This differs from rounding an IEEE addition toward zero.
inline float add_quad_offsets(float center, float dx, float dy) {
  const float largest = std::max({std::abs(center), std::abs(dx), std::abs(dy)});
  if (largest == 0)
    return 0;
  const double unit = std::ldexp(1.0, std::ilogb(largest) - 23);
  return truncate_float(
      (std::trunc(center / unit) + std::trunc(dx / unit) + std::trunc(dy / unit)) * unit);
}

struct Plane {
  float dx, dy;
  float base = 0;

  float at_quad(double x, double y, uint32_t lane) const {
    const float first = truncate_float(x * dx + y * dy + base);
    return add_quad_offsets(first, lane & 1 ? dx : 0, lane & 2 ? dy : 0);
  }
};

inline double round_subpixel(double value) {
  const double scaled = value * 256;
  const double lo = std::floor(scaled);
  const double fraction = scaled - lo;
  return (lo + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lo, 2.0) != 0))) / 256;
}

// Viewport scaling precedes perspective division. The multiplier truncates
// using its input exponents and retains a carry bit before normalization.
// Narrowing this intermediate to float loses that bit and changes coverage.
inline double multiply_viewport_scale(float position, float scale) {
  const double product = double(position) * scale;
  if (position == 0 || scale == 0 || !std::isfinite(product))
    return product;
  const double unit = std::ldexp(1.0, std::ilogb(position) + std::ilogb(scale) - 23);
  return std::trunc(product / unit) * unit;
}

// Projection and translation each truncate to single precision before the
// fixed-point conversion. Dividing position by W before scaling changes both
// the order of operations and the precision of the scale product.
inline double viewport_coordinate(float position, float w, float scale, float offset) {
  const double scaled = multiply_viewport_scale(position, scale);
  const float projected = truncate_float(scaled * reciprocal(w));
  return round_subpixel(truncate_float(double(projected) + offset));
}

} // namespace rocjitsu::amdgpu::raster
