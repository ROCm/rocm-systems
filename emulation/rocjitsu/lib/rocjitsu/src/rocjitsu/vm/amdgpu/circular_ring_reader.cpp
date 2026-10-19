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

VmAccessOutcome CircularRingReader::read(const GpuVmAccess &access, uint64_t cursor,
                                         std::span<std::byte> bytes,
                                         CircularRingReadProgress &progress,
                                         const VmTransferIngress &ingress) const {
  if (ring_bytes_ == 0 || ring_base_ > std::numeric_limits<uint64_t>::max() - (ring_bytes_ - 1) ||
      bytes.size() > ring_bytes_ || progress.completed_bytes > bytes.size() ||
      bytes.size() > std::numeric_limits<uint64_t>::max() - cursor)
    return VmAccessOutcome::Malformed;

  while (progress.completed_bytes < bytes.size()) {
    if (progress.segment_bytes == 0) {
      progress.segment_begin = progress.completed_bytes;
      const uint64_t ring_offset = (cursor + progress.segment_begin) % ring_bytes_;
      progress.segment_bytes =
          std::min<std::size_t>(bytes.size() - progress.segment_begin,
                                static_cast<std::size_t>(ring_bytes_ - ring_offset));
    }
    if (progress.segment_begin > bytes.size() ||
        progress.segment_bytes > bytes.size() - progress.segment_begin ||
        progress.segment_progress > progress.segment_bytes)
      return VmAccessOutcome::Malformed;
    const uint64_t ring_offset = (cursor + progress.segment_begin) % ring_bytes_;
    const VmAccessOutcome outcome = access.read(
        ring_base_ + ring_offset, bytes.subspan(progress.segment_begin, progress.segment_bytes),
        progress.segment_progress, progress.transfer_token, VmAccessKind::Read, true, {}, ingress);
    progress.completed_bytes = progress.segment_begin + progress.segment_progress;
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    if (progress.segment_progress != progress.segment_bytes)
      return VmAccessOutcome::Malformed;
    progress.segment_begin = progress.completed_bytes;
    progress.segment_bytes = 0;
    progress.segment_progress = 0;
  }
  return VmAccessOutcome::Complete;
}

} // namespace rocjitsu::amdgpu
