// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file aql_packet_types.h
/// @brief Protocol-level constants and normalized fields shared by AQL processing and CP state.

#pragma once

#include <cstddef>
#include <cstdint>

namespace rocjitsu::amdgpu {

inline constexpr std::size_t kAqlPacketBytes = 64;

/// @brief Normalized cluster dimensions decoded from an extended AQL dispatch packet.
struct ClusterDispatchShape {
  uint32_t count_x = 0;
  uint32_t count_y = 0;
  uint32_t count_z = 0;
  uint32_t size_x = 1;
  uint32_t size_y = 1;
  uint32_t size_z = 1;
};

} // namespace rocjitsu::amdgpu
