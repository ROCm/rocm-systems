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

} // namespace rocjitsu::amdgpu::raster
