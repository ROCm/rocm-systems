// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/prt_residency.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <span>

namespace rocjitsu::amdgpu {

struct MemoryByteRange {
  uint64_t address;
  uint32_t bytes;
};

// Visit only participating elements of a lane or sample tap, splitting swizzled
// elements at the same unit boundaries as the memory pipeline.
template <typename Visit>
bool any_texel_byte_range(const VectorMemState &d, uint32_t lane, uint64_t base,
                          const ElementLaneMasks &element_lanes, Visit visit) {
  const uint64_t lane_bit = uint64_t{1} << lane;
  const bool swizzled = d.scratch_swizzle && (d.scratch_lane_mask & lane_bit);
  const uint32_t unit = d.scratch_swizzle_unit;
  const uint32_t first_byte = swizzled ? (base - d.scratch_addr_base_offset) % unit : 0;
  if (!swizzled && element_lanes.empty())
    return visit(MemoryByteRange{base, d.num_elems * d.elem_size});
  for (uint32_t elem = 0; elem < d.num_elems; ++elem) {
    if (!element_lanes.empty() && !(element_lanes[elem] & lane_bit))
      continue;
    uint32_t copied = elem * d.elem_size;
    const uint32_t end = copied + d.elem_size;
    while (copied < end) {
      const uint32_t logical = first_byte + copied;
      const uint64_t address = swizzled ? base - first_byte +
                                              uint64_t{logical / unit} * d.scratch_addr_stride +
                                              logical % unit
                                        : base + copied;
      const uint32_t size = swizzled ? std::min(end - copied, unit - logical % unit) : end - copied;
      if (visit(MemoryByteRange{address, size}))
        return true;
      copied += size;
    }
  }
  return false;
}

inline bool overlaps_nonresident_prt(MemoryByteRange access, const PrtRanges &ranges) {
  const auto it = std::ranges::upper_bound(ranges, access.address, {}, &PrtRange::end);
  return it != ranges.end() && access.bytes != 0 &&
         (it->begin <= access.address || it->begin - access.address < access.bytes);
}

inline uint64_t nonresident_texel_lanes(const VectorMemState &d,
                                        std::span<const uint64_t> addresses, uint64_t lanes,
                                        const PrtRanges *ranges) {
  if (!ranges || ranges->empty())
    return 0;
  const auto overlaps = [&](MemoryByteRange access) {
    return overlaps_nonresident_prt(access, *ranges);
  };
  uint64_t missing = 0;
  while (lanes) {
    const auto lane = std::countr_zero(lanes);
    lanes &= lanes - 1;
    if (any_texel_byte_range(d, lane, addresses[lane], d.element_lane_masks, overlaps))
      missing |= uint64_t{1} << lane;
  }
  return missing;
}

} // namespace rocjitsu::amdgpu
