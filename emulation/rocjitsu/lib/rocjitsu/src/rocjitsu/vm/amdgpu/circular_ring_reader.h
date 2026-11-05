// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file circular_ring_reader.h
/// @brief Resumable reads from a byte-addressed circular GPU ring.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rocjitsu::amdgpu {

/// @brief Reads a logical byte range through the wrap point of a GPU ring.
/// @details The caller retains @p completed_bytes across Unavailable results.
/// Progress advances only for physical reads that completed, preventing replay
/// of an already-fetched prefix on retry.
class CircularRingReader {
public:
  CircularRingReader(uint64_t ring_base, uint64_t ring_bytes)
      : ring_base_(ring_base), ring_bytes_(ring_bytes) {}

  [[nodiscard]] VmAccessOutcome read(const GpuVmAccess &access, uint64_t cursor,
                                     std::span<std::byte> bytes,
                                     std::size_t &completed_bytes) const;
  [[nodiscard]] VmAccessOutcome read(const GpuVmAccess &access, uint64_t cursor,
                                     std::span<std::byte> bytes) const;

  [[nodiscard]] uint64_t ring_base() const { return ring_base_; }
  [[nodiscard]] uint64_t ring_bytes() const { return ring_bytes_; }

private:
  uint64_t ring_base_ = 0;
  uint64_t ring_bytes_ = 0;
};

} // namespace rocjitsu::amdgpu
