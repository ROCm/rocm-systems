// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file private_segment_allocator.h
/// @brief Shared per-lane private-segment range allocation primitive.

#pragma once

#include "util/bit.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace rocjitsu {

/// @brief Monotonic byte-range allocator for one kernel's private segment.
///
/// @details This deliberately knows nothing about register identity or spill
/// lifetime. DBI layers stable register-to-slot mapping on top; semantic DBT
/// creates short-lived frames on top. Sharing only range arithmetic keeps those
/// different policies separate while guaranteeing identical alignment and
/// overflow behavior.
class PrivateSegmentCursor final {
public:
  /// @brief Begin allocation at the first byte not owned by an earlier policy.
  explicit PrivateSegmentCursor(uint32_t first_byte) : cursor_(first_byte) {}

  /// @brief Compute the next aligned allocation without advancing the cursor.
  /// @param byte_count Number of contiguous bytes requested.
  /// @param alignment Required byte alignment for the returned base.
  /// @param limit Exclusive upper bound for the allocation.
  /// @returns The aligned base, or std::nullopt for an invalid or overflowing range.
  [[nodiscard]] std::optional<uint32_t>
  preview(uint32_t byte_count, uint32_t alignment,
          uint32_t limit = std::numeric_limits<uint32_t>::max()) const {
    if (byte_count == 0 || alignment == 0)
      return std::nullopt;
    const uint64_t base =
        util::align_up(static_cast<uint64_t>(cursor_), static_cast<uint64_t>(alignment));
    if (base + byte_count > limit)
      return std::nullopt;
    return static_cast<uint32_t>(base);
  }

  /// @brief Allocate the next aligned range and advance the high-water mark.
  /// @param byte_count Number of contiguous bytes requested.
  /// @param alignment Required byte alignment for the returned base.
  /// @param limit Exclusive upper bound for the allocation.
  /// @returns The allocated base, or std::nullopt without changing state.
  [[nodiscard]] std::optional<uint32_t>
  allocate(uint32_t byte_count, uint32_t alignment,
           uint32_t limit = std::numeric_limits<uint32_t>::max()) {
    auto base = preview(byte_count, alignment, limit);
    if (!base)
      return std::nullopt;
    cursor_ = *base + byte_count;
    return base;
  }

  /// @brief One-past-end byte of all successfully allocated ranges.
  [[nodiscard]] uint32_t high_water_mark() const { return cursor_; }

private:
  uint32_t cursor_ = 0;
};

} // namespace rocjitsu
