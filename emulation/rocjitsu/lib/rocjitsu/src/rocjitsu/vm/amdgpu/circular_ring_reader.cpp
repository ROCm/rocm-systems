// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/circular_ring_reader.h"

#include <algorithm>
#include <limits>

namespace rocjitsu::amdgpu {

VmAccessOutcome CircularRingReader::read(const GpuVmAccess &access, uint64_t cursor,
                                         std::span<std::byte> bytes,
                                         std::size_t &completed_bytes) const {
  if (ring_bytes_ == 0 || ring_base_ > std::numeric_limits<uint64_t>::max() - (ring_bytes_ - 1) ||
      bytes.size() > ring_bytes_ || completed_bytes > bytes.size() ||
      bytes.size() > std::numeric_limits<uint64_t>::max() - cursor) {
    return VmAccessOutcome::Malformed;
  }

  while (completed_bytes < bytes.size()) {
    const uint64_t ring_offset = (cursor + completed_bytes) % ring_bytes_;
    const std::size_t chunk = std::min<std::size_t>(
        bytes.size() - completed_bytes, static_cast<std::size_t>(ring_bytes_ - ring_offset));
    std::size_t segment_progress = 0;
    const VmAccessOutcome outcome = access.read(
        ring_base_ + ring_offset, bytes.subspan(completed_bytes, chunk), segment_progress);
    completed_bytes += segment_progress;
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    if (segment_progress != chunk)
      return VmAccessOutcome::Malformed;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome CircularRingReader::read(const GpuVmAccess &access, uint64_t cursor,
                                         std::span<std::byte> bytes) const {
  std::size_t completed_bytes = 0;
  return read(access, cursor, bytes, completed_bytes);
}

} // namespace rocjitsu::amdgpu
