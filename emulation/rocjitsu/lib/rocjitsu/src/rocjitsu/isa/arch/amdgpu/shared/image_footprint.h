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

// The fixed-point model uses 1/256 LOD units. A zero gradient selects -32
// before the sampler applies its minimum LOD clamp.
inline constexpr int32_t kImageLodFractionScale = 256;
inline constexpr int32_t kZeroGradientLodFixed = -32 * kImageLodFractionScale;

/// @brief Fixed-point sampler norm qualified on GFX11/12.
/// @details Five projections replace a Euclidean norm. Intermediate norm carry
/// bits affect whole-mip selection and must survive until logarithm conversion.
// The five projection pairs and eight logarithm segments below come from
// the Vulkan footprint model (image_polygon_norm and the derivative logarithm):
// https://github.com/ROCm/rocm-systems/blob/75385e52eeb8a511eac26643f3efee0245c3e551/emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/graphics/image_filter.h
// These are the model's coefficients, empirically qualified against gfx1100
// and gfx1201 captures. This HIP path fits no additional coefficients.
inline uint32_t image_polygon_norm(int32_t first_component, int32_t second_component) {
  const uint32_t major_magnitude = std::max(std::abs(first_component), std::abs(second_component));
  const uint32_t minor_magnitude = std::min(std::abs(first_component), std::abs(second_component));
  const uint32_t projection = std::max(
      {256 * major_magnitude + 16 * minor_magnitude, 250 * major_magnitude + 56 * minor_magnitude,
       238 * major_magnitude + 96 * minor_magnitude, 221 * major_magnitude + 130 * minor_magnitude,
       197 * major_magnitude + 164 * minor_magnitude});
  return (projection + 128) >> 8;
}

/// @brief Convert an aligned footprint norm to signed 1/256 LOD units.
inline int32_t image_footprint_logarithm(uint32_t footprint_norm, int exponent) {
  if (footprint_norm == 0)
    return kZeroGradientLodFixed;
  const uint32_t leading_bit = std::bit_width(footprint_norm) - 1;
  const uint32_t denominator = 1u << leading_bit;
  const uint32_t fraction = (footprint_norm - denominator) * 1024;
  const uint32_t segment = fraction / (128 * denominator);
  constexpr std::array<uint32_t, 8> logarithm_bases{0, 175, 330, 471, 599, 717, 827, 929};
  constexpr std::array<uint32_t, 8> logarithm_slopes{176, 155, 142, 129, 119, 110, 102, 95};
  const uint32_t logarithm =
      (logarithm_bases[segment] * 128 * denominator +
       logarithm_slopes[segment] * (fraction - segment * 128 * denominator) + 256 * denominator) /
      (512 * denominator);
  return (exponent - 11 + static_cast<int>(leading_bit)) * kImageLodFractionScale +
         static_cast<int32_t>(logarithm);
}

/// @brief Derive sampler LOD from raw gradients and view-relative image extents.
/// @details Align each coordinate's screen derivatives before multiplying by
/// its extent, then align coordinates for the polygonal norms. Conversion ties
/// round away from zero; multiplication and alignment ties round upward.
/// 1D/2D use the principal-axis construction; volumes use the larger of two
/// nested XY/Z vector norms. GFX11/12 captures qualify the precision stages;
/// older image architectures use the same model without physical qualification.
inline double image_gradient_lod(std::array<double, 3> screen_dx, std::array<double, 3> screen_dy,
                                 const std::array<uint32_t, 3> &extents, uint32_t dimensions) {
  std::array<double, 6> gradients{screen_dx[0], screen_dx[1], screen_dx[2],
                                  screen_dy[0], screen_dy[1], screen_dy[2]};
  std::array<double, 3> largest_gradient{};
  for (uint32_t coordinate = 0; coordinate < dimensions; ++coordinate) {
    for (uint32_t gradient_index : {coordinate, coordinate + 3}) {
      if (std::isnan(gradients[gradient_index]))
        gradients[gradient_index] = 0;
      else if (std::isinf(gradients[gradient_index]))
        gradients[gradient_index] =
            std::copysign(double(std::numeric_limits<float>::max()), gradients[gradient_index]);
    }
    largest_gradient[coordinate] =
        std::max(std::abs(gradients[coordinate]), std::abs(gradients[coordinate + 3]));
  }
  if (std::all_of(largest_gradient.begin(), largest_gradient.end(),
                  [](double magnitude) { return magnitude == 0; }))
    return double(kZeroGradientLodFixed) / kImageLodFractionScale;
  std::array<int, 3> exponents{};
  std::array<double, 6> scaled{};
  int exponent = std::numeric_limits<int>::min();
  for (uint32_t coordinate = 0; coordinate < dimensions; ++coordinate) {
    if (largest_gradient[coordinate] == 0)
      continue;
    int coordinate_exponent;
    std::frexp(largest_gradient[coordinate], &coordinate_exponent);
    const int extent_exponent = std::bit_width(extents[coordinate]) - 1;
    const double extent_mantissa =
        std::floor(std::ldexp(double(extents[coordinate]), 10 - extent_exponent)) / 1024;
    exponents[coordinate] = coordinate_exponent + extent_exponent;
    exponent = std::max(exponent, exponents[coordinate]);
    for (uint32_t gradient_index : {coordinate, coordinate + 3}) {
      const double significand = std::min(
          2047.0,
          std::floor(std::ldexp(std::abs(gradients[gradient_index]), 11 - coordinate_exponent) +
                     0.5));
      scaled[gradient_index] =
          std::floor(std::copysign(significand, gradients[gradient_index]) * extent_mantissa + 0.5);
    }
  }
  std::array<int32_t, 6> fixed{};
  for (uint32_t coordinate = 0; coordinate < dimensions; ++coordinate) {
    if (largest_gradient[coordinate] == 0)
      continue;
    for (uint32_t gradient_index : {coordinate, coordinate + 3})
      fixed[gradient_index] = static_cast<int32_t>(
          std::floor(std::ldexp(scaled[gradient_index], exponents[coordinate] - exponent) + 0.5));
  }
  uint32_t footprint_norm;
  if (dimensions == 3) {
    footprint_norm = std::max(image_polygon_norm(image_polygon_norm(fixed[0], fixed[1]), fixed[2]),
                              image_polygon_norm(image_polygon_norm(fixed[3], fixed[4]), fixed[5]));
  } else {
    const uint32_t first_norm = image_polygon_norm(fixed[0] + fixed[4], fixed[1] - fixed[3]);
    const uint32_t second_norm = image_polygon_norm(fixed[0] - fixed[4], fixed[1] + fixed[3]);
    footprint_norm = (first_norm + second_norm) >> 1;
  }
  return image_footprint_logarithm(footprint_norm, exponent) / double(kImageLodFractionScale);
}

} // namespace rocjitsu::amdgpu
