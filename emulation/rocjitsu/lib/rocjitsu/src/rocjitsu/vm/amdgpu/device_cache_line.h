// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocjitsu::amdgpu {

inline constexpr uint32_t kDeviceCacheLineBits = 7;
inline constexpr uint32_t kDeviceCacheLineBytes = 1U << kDeviceCacheLineBits;

} // namespace rocjitsu::amdgpu
