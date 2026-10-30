// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file queue_doorbell.h
/// @brief Transport-independent queue notification modes.

#pragma once

#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief How a queue owner learns that its producer cursor advanced.
enum class QueueDoorbellMode : uint8_t {
  /// Poll a host-mapped doorbell page and clamp work to the observed cursor.
  HostPolled,
  /// Poll a doorbell value through the queue's GPU virtual address space.
  VmPolled,
  /// Rely solely on explicit frontend notifications and clamp to their cursor.
  Explicit,
};

} // namespace rocjitsu::amdgpu
