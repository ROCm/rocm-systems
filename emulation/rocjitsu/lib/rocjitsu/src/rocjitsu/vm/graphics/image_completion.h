// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocjitsu::amdgpu {
class Wavefront;
class ComputeUnitCore;
class VectorMemState;

/// Decode and filter prepared texel responses, then publish shader results.
void complete_image_load(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &state);
} // namespace rocjitsu::amdgpu
