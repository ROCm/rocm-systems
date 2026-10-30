// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <utility>

namespace rocjitsu::amdgpu {

/// @brief First-fit LDS reservations, represented only by address-ordered free ranges.
/// @details Map keys are byte offsets; values are lengths. Does not own or clear LDS data. The CU
/// owns synchronization and decides when workgroup/cluster lifetimes permit a reservation to be
/// released.
class LdsAllocator {
public:
  static constexpr uint32_t kAlignment = 256;

  explicit LdsAllocator(uint32_t capacity) : capacity_(capacity) { reset(); }

  /// @brief Whether count separate, equally sized reservations fit without mutation.
  bool can_allocate(uint32_t requested_bytes, uint32_t count = 1) const {
    const uint64_t bytes = aligned_size(requested_bytes);
    if (bytes == 0 || count == 0)
      return true;
    uint64_t available = 0;
    for (const auto &range : free_ranges_) {
      available += range.second / bytes;
      if (available >= count)
        return true;
    }
    return false;
  }

  /// @brief Reserve an aligned contiguous range, or leave all state unchanged.
  /// @details A zero-byte request succeeds at offset zero without reserving space.
  [[nodiscard]] std::optional<uint32_t> allocate(uint32_t requested_bytes) {
    const uint64_t bytes = aligned_size(requested_bytes);
    if (bytes == 0)
      return 0;
    const auto candidate = std::ranges::find_if(
        free_ranges_, [bytes](const auto &free_range) { return free_range.second >= bytes; });
    if (candidate == free_ranges_.end())
      return std::nullopt;
    auto node = free_ranges_.extract(candidate);
    const uint32_t offset = node.key();
    if (node.mapped() > bytes) {
      node.key() += static_cast<uint32_t>(bytes);
      node.mapped() -= static_cast<uint32_t>(bytes);
      free_ranges_.insert(std::move(node));
    }
    return offset;
  }

  /// @brief Return one live reservation, using the original requested size.
  /// @details Call exactly once per reservation, after its users are quiescent.
  /// Adjacent free ranges are coalesced. A zero-byte release is a no-op.
  void release(uint32_t offset, uint32_t requested_bytes) {
    const uint64_t bytes = aligned_size(requested_bytes);
    if (bytes == 0)
      return;
    const uint64_t end = uint64_t{offset} + bytes;
    assert(offset % kAlignment == 0 && end <= capacity_);
    const auto next = free_ranges_.lower_bound(offset);
    assert(next == free_ranges_.end() || end <= next->first);
    if (next != free_ranges_.begin()) {
      const auto previous = std::prev(next);
      const uint64_t previous_end = uint64_t{previous->first} + previous->second;
      assert(previous_end <= offset);
      if (previous_end == offset) {
        previous->second += static_cast<uint32_t>(bytes);
        if (next != free_ranges_.end() && end == next->first) {
          previous->second += next->second;
          free_ranges_.erase(next);
        }
        return;
      }
    }
    if (next != free_ranges_.end() && end == next->first) {
      auto node = free_ranges_.extract(next);
      node.key() = offset;
      node.mapped() += static_cast<uint32_t>(bytes);
      free_ranges_.insert(std::move(node));
    } else {
      free_ranges_.emplace_hint(next, offset, static_cast<uint32_t>(bytes));
    }
  }

  /// @brief Reclaim the complete pool after all users and cluster pins are gone.
  void reset() { free_ranges_ = {{0, capacity_}}; }

private:
  static uint64_t aligned_size(uint32_t bytes) {
    return (uint64_t{bytes} + kAlignment - 1) / kAlignment * kAlignment;
  }

  uint32_t capacity_;
  std::map<uint32_t, uint32_t> free_ranges_;
};

} // namespace rocjitsu::amdgpu
