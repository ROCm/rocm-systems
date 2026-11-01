// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu::amdgpu {

/// @brief Fixed-point sampler norm qualified on GFX11/12.
/// @details Five projections replace a Euclidean norm. Intermediate norm carry
/// bits affect whole-mip selection and must survive until logarithm conversion.
inline uint32_t image_polygon_norm(int32_t x, int32_t y) {
  const uint32_t a = std::max(std::abs(x), std::abs(y));
  const uint32_t b = std::min(std::abs(x), std::abs(y));
  const uint32_t projection = std::max(
      {256 * a + 16 * b, 250 * a + 56 * b, 238 * a + 96 * b, 221 * a + 130 * b, 197 * a + 164 * b});
  return (projection + 128) >> 8;
}

/// @brief Convert an aligned footprint norm to signed 1/256 LOD units.
inline int32_t image_footprint_logarithm(uint32_t axis, int exponent) {
  if (axis == 0)
    return -8192;
  const uint32_t leading = std::bit_width(axis) - 1;
  const uint32_t denominator = 1u << leading;
  const uint32_t fraction = (axis - denominator) * 1024;
  const uint32_t segment = fraction / (128 * denominator);
  constexpr std::array<uint32_t, 8> bases{0, 175, 330, 471, 599, 717, 827, 929};
  constexpr std::array<uint32_t, 8> slopes{176, 155, 142, 129, 119, 110, 102, 95};
  const uint32_t logarithm =
      (bases[segment] * 128 * denominator +
       slopes[segment] * (fraction - segment * 128 * denominator) + 256 * denominator) /
      (512 * denominator);
  return (exponent - 11 + static_cast<int>(leading)) * 256 + static_cast<int32_t>(logarithm);
}

/// @brief Derive sampler LOD from raw gradients and view-relative image extents.
/// @details Align each coordinate's screen derivatives before multiplying by
/// its extent, then align coordinates for the polygonal norms. Conversion ties
/// round away from zero; multiplication and alignment ties round upward.
/// 1D/2D use the principal-axis construction; volumes use the larger of two
/// nested XY/Z vector norms. GFX11/12 captures qualify the precision stages;
/// older image architectures use the same model without physical qualification.
inline double image_gradient_lod(std::array<double, 3> dx, std::array<double, 3> dy,
                                 const std::array<uint32_t, 3> &extents, uint32_t dimensions) {
  std::array<double, 6> gradients{dx[0], dx[1], dx[2], dy[0], dy[1], dy[2]};
  std::array<double, 3> largest{};
  for (uint32_t axis = 0; axis < dimensions; ++axis) {
    for (uint32_t i : {axis, axis + 3}) {
      if (std::isnan(gradients[i]))
        gradients[i] = 0;
      else if (std::isinf(gradients[i]))
        gradients[i] = std::copysign(double(std::numeric_limits<float>::max()), gradients[i]);
    }
    largest[axis] = std::max(std::abs(gradients[axis]), std::abs(gradients[axis + 3]));
  }
  if (std::all_of(largest.begin(), largest.end(), [](double x) { return x == 0; }))
    return -32;
  std::array<int, 3> exponents{};
  std::array<double, 6> scaled{};
  int exponent = std::numeric_limits<int>::min();
  for (uint32_t axis = 0; axis < dimensions; ++axis) {
    if (largest[axis] == 0)
      continue;
    int coordinate_exponent;
    std::frexp(largest[axis], &coordinate_exponent);
    const int extent_exponent = std::bit_width(extents[axis]) - 1;
    const double extent_mantissa =
        std::floor(std::ldexp(double(extents[axis]), 10 - extent_exponent)) / 1024;
    exponents[axis] = coordinate_exponent + extent_exponent;
    exponent = std::max(exponent, exponents[axis]);
    for (uint32_t i : {axis, axis + 3}) {
      const double significand = std::min(
          2047.0, std::floor(std::ldexp(std::abs(gradients[i]), 11 - coordinate_exponent) + 0.5));
      scaled[i] = std::floor(std::copysign(significand, gradients[i]) * extent_mantissa + 0.5);
    }
  }
  std::array<int32_t, 6> fixed{};
  for (uint32_t axis = 0; axis < dimensions; ++axis) {
    if (largest[axis] == 0)
      continue;
    for (uint32_t i : {axis, axis + 3})
      fixed[i] =
          static_cast<int32_t>(std::floor(std::ldexp(scaled[i], exponents[axis] - exponent) + 0.5));
  }
  uint32_t major;
  if (dimensions == 3) {
    major = std::max(image_polygon_norm(image_polygon_norm(fixed[0], fixed[1]), fixed[2]),
                     image_polygon_norm(image_polygon_norm(fixed[3], fixed[4]), fixed[5]));
  } else {
    const uint32_t first = image_polygon_norm(fixed[0] + fixed[4], fixed[1] - fixed[3]);
    const uint32_t second = image_polygon_norm(fixed[0] - fixed[4], fixed[1] + fixed[3]);
    major = (first + second) >> 1;
  }
  return image_footprint_logarithm(major, exponent) / 256.0;
}

} // namespace rocjitsu::amdgpu
