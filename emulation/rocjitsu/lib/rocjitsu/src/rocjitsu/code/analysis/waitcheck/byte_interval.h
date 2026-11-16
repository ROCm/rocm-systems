// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>

namespace rocjitsu::waitcheck_detail {

// Half-open byte offsets from a single proven base in a bounded memory segment.
struct ByteInterval {
  uint32_t begin, end;

  static constexpr std::optional<ByteInterval> within(uint32_t offset, uint32_t width,
                                                      uint32_t segment_size) {
    if (width == 0 || offset > segment_size || width > segment_size - offset)
      return std::nullopt;
    return ByteInterval{offset, offset + width};
  }
};

enum class IntervalRelation { Unknown, Disjoint, Identical, Overlap };

constexpr IntervalRelation compare_intervals(std::optional<ByteInterval> a,
                                             std::optional<ByteInterval> b) {
  if (!a || !b)
    return IntervalRelation::Unknown;
  if (a->begin == b->begin && a->end == b->end)
    return IntervalRelation::Identical;
  if (a->end <= b->begin || b->end <= a->begin)
    return IntervalRelation::Disjoint;
  return IntervalRelation::Overlap;
}

} // namespace rocjitsu::waitcheck_detail
