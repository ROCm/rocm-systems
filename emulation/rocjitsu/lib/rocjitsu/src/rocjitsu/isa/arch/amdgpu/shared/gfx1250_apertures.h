// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocjitsu::amdgpu {

// GFX1250 flat private pointers use a fixed 57-bit window. These are the
// device defaults for standalone simulation and the windows advertised by KFD.
inline constexpr uint64_t kGfx1250PrivateApertureBase = 0x1000'0000'0000'0000ULL;
inline constexpr uint64_t kGfx1250PrivateApertureLimit =
    kGfx1250PrivateApertureBase + (uint64_t{1} << 57) - 1;
inline constexpr uint64_t kGfx1250SharedApertureBase = 0x2000'0000'0000'0000ULL;
inline constexpr uint64_t kGfx1250SharedApertureLimit = kGfx1250SharedApertureBase + UINT32_MAX;

} // namespace rocjitsu::amdgpu
