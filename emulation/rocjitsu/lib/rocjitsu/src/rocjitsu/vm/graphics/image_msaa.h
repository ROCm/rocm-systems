// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocjitsu::amdgpu {

/// Byte address in an uncompressed GFX11 Z_X or R_X 2D multisample surface.
/// pitch and height are in texels. The caller validates all selectors.
uint64_t gfx11_msaa_address(uint64_t base, uint32_t x, uint32_t y, uint32_t layer, uint32_t sample,
                            uint32_t pitch, uint32_t height, uint32_t bytes, uint32_t sample_log2,
                            uint32_t swizzle);

} // namespace rocjitsu::amdgpu
