// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_handles.h
/// @brief Lifetime-safe identities shared by GPU core services and front ends.

#pragma once

#include <cstdint>
#include <limits>

namespace rocjitsu::amdgpu {

/// @brief Lifetime-safe reference to one GPU address space.
struct AddressSpaceHandle {
  static constexpr uint32_t kInvalidSlot = std::numeric_limits<uint32_t>::max();

  uint32_t slot = kInvalidSlot;
  uint64_t generation = 0;

  explicit operator bool() const { return slot != kInvalidSlot && generation != 0; }
  friend bool operator==(const AddressSpaceHandle &, const AddressSpaceHandle &) = default;
};

/// @brief Lifetime-safe reference to one registered hardware queue.
struct QueueHandle {
  static constexpr uint32_t kInvalidSlot = std::numeric_limits<uint32_t>::max();

  uint32_t slot = kInvalidSlot;
  uint64_t generation = 0;

  explicit operator bool() const { return slot != kInvalidSlot && generation != 0; }
  friend bool operator==(const QueueHandle &, const QueueHandle &) = default;
};

} // namespace rocjitsu::amdgpu
