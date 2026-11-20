// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu {
class Wavefront;
class ComputeUnitCore;
class VectorMemState;

/// Filter one four-texel footprint at the texture unit's shared-exponent precision.
double filter_image_float_texels(std::array<double, 4> channels,
                                 const std::array<double, 4> &weights, uint32_t unaligned,
                                 uint32_t precision);

/// Round interpolated hardware-filter results before the final FP32 conversion.
double round_image_filter_result(double filtered, int exponent, bool fp32);

/// Decode and filter prepared texel responses, then publish shader results.
void complete_image_load(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &state);
} // namespace rocjitsu::amdgpu
