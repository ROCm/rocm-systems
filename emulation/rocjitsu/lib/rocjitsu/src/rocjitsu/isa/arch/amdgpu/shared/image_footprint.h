// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/detail/image_footprint.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rocjitsu::amdgpu {

// The fixed-point model uses 1/256 LOD units. The isotropic footprint uses
// -32 for zero gradients before sampler LOD clamping.
inline constexpr int32_t kImageLodFractionScale = 256;
inline constexpr int32_t kZeroGradientLodFixed = -32 * kImageLodFractionScale;
// On gfx1100 and gfx1201 anisotropic LOD bottoms out at -64 before sampler bias.
inline constexpr int32_t kMinimumAnisotropicLodFixed = -64 * kImageLodFractionScale;

inline int32_t image_anisotropic_lod(int32_t major_lod, int32_t minor_lod,
                                     uint32_t max_anisotropy) {
  return std::max({kMinimumAnisotropicLodFixed, minor_lod,
                   major_lod - std::countr_zero(max_anisotropy) * kImageLodFractionScale});
}

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

/// @brief Derive sampler LOD from raw gradients and view-relative image extents.
/// @details Align each coordinate's screen derivatives before multiplying by
/// its extent, then align coordinates for the polygonal norms. Conversion ties
/// round away from zero; multiplication and alignment ties round upward.
/// 1D/2D use the principal-axis construction; volumes use the larger of two
/// nested XY/Z vector norms. GFX11/12 captures qualify the precision stages;
/// older image architectures use the same model without physical qualification.
inline double image_gradient_lod(std::array<double, 3> screen_dx, std::array<double, 3> screen_dy,
                                 const std::array<uint32_t, 3> &extents, uint32_t dimensions,
                                 uint32_t max_anisotropy = 0, uint64_t cpu_flags = 0) {
  rj_arith_image_gradients gradients{};
  for (int i = 0; i < 3; ++i) {
    gradients.dx[i] = screen_dx[i];
    gradients.dy[i] = screen_dy[i];
  }
  rj_arith_image_footprint result;
  rj_arith_detail_image_footprint(cpu_flags, &gradients, extents.data(), dimensions, &result);
  const bool zero = result.norms[0] == 0 && result.norms[1] == 0;
  const int32_t lod =
      max_anisotropy && dimensions < 3
          ? zero ? kMinimumAnisotropicLodFixed
                 : image_anisotropic_lod(result.lods[0], result.lods[1], max_anisotropy)
          : result.lods[0];
  return lod / double(kImageLodFractionScale);
}

} // namespace rocjitsu::amdgpu
